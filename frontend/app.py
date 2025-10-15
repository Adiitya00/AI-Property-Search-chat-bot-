import streamlit as st
import requests

# --- Page config ---
st.set_page_config(
    page_title="NoBrokerage AI Property Search",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Initialize session state ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# --- API query function ---
def query_api(query: str, api_url: str):
    try:
        last_city = None
        for msg in reversed(st.session_state.messages):
            if msg['role']=='assistant' and 'meta' in msg and 'appliedFilters' in msg['meta']:
                last_city = msg['meta']['appliedFilters'].get('city')
                break
        payload = {"query": query, "context": {"last_city": last_city}}
        response = requests.post(f"{api_url}/api/query", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# --- Display chat messages ---
def display_chat_message(message):
    if message['role'] == 'user':
        st.markdown(f'''
        <div style="
            background: linear-gradient(135deg,#667eea,#764ba2);
            color: white;
            padding:1rem 1.5rem;
            border-radius: 20px 20px 5px 20px;
            margin:0.8rem 0;
            max-width:85%;
            margin-left:auto;
            box-shadow:0 2px 8px rgba(102,126,234,0.3);
        ">
            {message["content"]}
        </div>
        ''', unsafe_allow_html=True)
    else:
        # Bot message summary
        st.markdown(f'''
        <div style="
            background: linear-gradient(135deg,#667eea,#764ba2);
            color: white;
            padding:1.2rem 1.5rem;
            border-radius:16px;
            margin:0.8rem 0;
            line-height:1.6;
        ">
            {message["content"].replace('\n','<br>')}
        </div>
        ''', unsafe_allow_html=True)

        # Display property cards if present
        if 'cards' in message and message['cards']:
            st.markdown("---")
            st.subheader("🏡 Matching Properties:")
            cards = message['cards']
            cols_per_row = 3
            for i in range(0, len(cards), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i+j < len(cards):
                        with col:
                            display_property_card(cards[i+j])

# --- Display property card ---
def display_property_card(card):
    possession = card.get('possession', 'Not specified')
    possession_color = "#eef2ff" if possession=="Not specified" else ("#d1fae5" if "ready" in possession.lower() else "#dbeafe")
    possession_text_color = "#667eea" if possession=="Not specified" else ("#065f46" if "ready" in possession.lower() else "#1e40af")
    
    amenities_html = ''.join([f'<span style="background:#eef2ff;color:#667eea;padding:0.3rem 0.6rem;border-radius:8px;font-size:0.85rem;margin:0.2rem;display:inline-block;">✨ {a}</span>' for a in card.get('amenities',[])])
    
    st.markdown(f'''
    <div style="
        background:white;
        border:1px solid #e5e7eb;
        border-radius:16px;
        padding:1rem;
        margin:0.5rem 0;
        box-shadow:0 2px 8px rgba(0,0,0,0.05);
        display:flex;
        flex-direction:column;
        height:100%;
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;font-weight:700;font-size:1.2rem;color:#1f2937;">
            {card['title']}
            <span style="background:{possession_color}; color:{possession_text_color}; padding:0.3rem 0.6rem; border-radius:12px;">{possession}</span>
        </div>
        <div style="color:#6b7280;font-size:0.95rem;margin:0.5rem 0;">📍 {card['city_locality']}</div>
        <div style="font-size:1.4rem;font-weight:700;color:#374151;">💰 {card['price']}</div>
        <div style="margin-top:0.5rem;">{amenities_html}</div>
    </div>
    ''', unsafe_allow_html=True)

# --- Main app ---
def main():
    st.markdown('<div style="font-family:Segoe UI, sans-serif;padding:1rem;">', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align:center;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🏠 NoBrokerage AI Property Search</h1>', unsafe_allow_html=True)
    
    # Input
    col1, col2 = st.columns([5,1])
    with col1:
        user_query = st.text_input("Ask me about properties...", placeholder="3BHK in Pune under 1.2 Cr")
    with col2:
        send_button = st.button("🚀 Search")
    
    # Display previous chat
    for msg in st.session_state.messages:
        display_chat_message(msg)
    
    # Process query
    if send_button and user_query:
        st.session_state.messages.append({"role":"user","content":user_query})
        with st.spinner("Searching..."):
            result = query_api(user_query, st.session_state.api_url)
        if result and not result.get('error'):
            summary = result.get('summary','No summary available.')
            cards = result.get('cards',[])
            st.session_state.messages.append({"role":"assistant","content":summary,"cards":cards})
        else:
            st.session_state.messages.append({"role":"assistant","content":f"Error: {result.get('error','Unknown')}"})
        st.experimental_rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
