import os
import re
from typing import Optional, List, Dict, Any
from collections import Counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Global stores populated during startup
merged_data: Optional[pd.DataFrame] = None
cities_list: List[str] = []
localities_list: List[str] = []

# --- FastAPI App Initialization ---
app = FastAPI(title="NoBrokerage AI Property Search API", version="2.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class PropertyCard(BaseModel):
    title: str
    city_locality: str
    bhk: str
    price: str
    project_name: str
    possession: str
    amenities: List[str]
    cta: str

class QueryResponse(BaseModel):
    summary: str
    cards: List[PropertyCard]
    meta: Dict[str, Any]

# -----------------------
# Helper functions
# -----------------------

def safe_title(s):
    """Safely title-case a string, handling NaN/empty/non-string inputs."""
    return str(s).strip().title() if pd.notna(s) and str(s).strip() else "Not Specified"

def parse_address_for_city_locality(address: str, cities: List[str]):
    """
    Parses a given address string to find a known city and locality.
    Prioritizes matching against a list of known cities.
    """
    if not address or not isinstance(address, str) or not address.strip():
        return "Not Specified", "Not Specified"

    # Normalize the address for matching
    norm_address = address.lower()
    
    # First, try to detect a known city
    detected_city = "Not Specified"
    for city in cities:
        if city.lower() in norm_address:
            detected_city = safe_title(city)
            break
            
    # Then, parse locality based on comma-separated parts
    parts = [p.strip() for p in address.split(",") if p.strip()]
    locality = "Not Specified"
    
    # If a city was found, try to find the part before it as the locality
    if detected_city != "Not Specified":
        try:
            city_index = parts.index(detected_city)
            if city_index > 0:
                locality = safe_title(parts[city_index - 1])
        except ValueError:
            # If city not in parts list, fall back to simple parsing
            if len(parts) >= 2:
                locality = safe_title(parts[-2])
    elif len(parts) >= 2:
        # Fallback for addresses without a detected city
        detected_city = safe_title(parts[-1])
        locality = safe_title(parts[-2])

    return detected_city, locality

def normalize_amenities_field(x):
    """
    Convert Amenities column into a list of strings.
    """
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    
    return [part.strip() for part in re.split(r',|\||;', s) if part.strip()]

def format_price(price_inr):
    """Format price into Lakhs (L) or Crores (Cr)."""
    if pd.isna(price_inr) or price_inr == 0:
        return "Price on request"
    try:
        price = float(price_inr)
    except Exception:
        return str(price_inr)
    if price >= 10000000:
        return f"₹{price/10000000:.2f} Cr"
    elif price >= 100000:
        return f"₹{price/100000:.2f} L"
    else:
        return f"₹{int(price):,}"

# -----------------------
# Natural Language Parser
# -----------------------
class NaturalLanguageParser:
    def __init__(self, cities: List[str], localities: List[str]):
        self.cities = [c.lower() for c in (cities or []) if c and c != 'not specified']
        self.localities = [l.lower() for l in (localities or []) if l and l != 'not specified']

        self.budget_patterns = [
            (r'between\s+₹?\s*([\d\.]+)\s*(?:and|to|-)\s*₹?\s*([\d\.]+)\s*(cr|crore|l|lac|lakh|k)?', 'range'),
            (r'under\s+₹?\s*([\d\.]+)\s*(cr|crore|l|lac|lakh|k)?', 'under'),
            (r'below\s+₹?\s*([\d\.]+)\s*(cr|crore|l|lac|lakh|k)?', 'under'),
            (r'up\s*to\s+₹?\s*([\d\.]+)\s*(cr|crore|l|lac|lakh|k)?', 'under'),
            (r'above\s+₹?\s*([\d\.]+)\s*(cr|crore|l|lac|lakh|k)?', 'above'),
            (r'around\s+₹?\s*([\d\.]+)\s*(cr|crore|l|lac|lakh|k)?', 'around'),
            (r'₹\s*([\d\.]+)\s*(cr|crore|l|lac|lakh|k)?', 'exact'),
        ]

        self.possession_map = {
            'ready': ['ready', 'ready to move', 'rtm', 'immediate', 'possession', 'available', 'move-in'],
            'under construction': ['under construction', 'uc', 'upcoming', 'launch', 'pre-launch', 'new launch', 'construction']
        }

        self.amenity_keywords = {
            'gym': ['gym', 'fitness', 'gymnasium'],
            'pool': ['pool', 'swimming'],
            'clubhouse': ['clubhouse', 'club house', 'club'],
            'parking': ['parking', 'car park'],
            'security': ['security', '24x7', 'gated'],
            'garden': ['garden', 'park', 'green'],
            'metro': ['metro', 'station'],
            'lift': ['lift', 'elevator']
        }

    def parse_price(self, num: float, unit: Optional[str]) -> float:
        if unit in ['cr', 'crore']:
            return num * 10000000
        if unit in ['l', 'lac', 'lakh']:
            return num * 100000
        if unit in ['k', 'thousand']:
            return num * 1000
        return num * 100000 if num < 1000 else num

    def extract_city(self, query: str) -> Optional[str]:
        q = query.lower()
        for city in self.cities:
            if re.search(r'\b' + re.escape(city) + r'\b', q) or (city in q):
                return city.title()
        return None

    def extract_locality(self, query: str, city: Optional[str]) -> Optional[str]:
        q = query.lower()
        if city:
            q = q.replace(city.lower(), '')
        for loc in self.localities:
            if re.search(r'\b' + re.escape(loc) + r'\b', q) or (loc in q):
                return loc.title()
        
        m = re.search(r'(?:in|near|at)\s+([a-zA-Z0-9\s\-]+)', q)
        if m:
            pot = m.group(1).strip()
            for loc in self.localities:
                if loc in pot or pot in loc:
                    return loc.title()
        return None

    def extract_bhk(self, query: str) -> Optional[int]:
        m = re.search(r'(\d+)\s*(bhk|bed|bedroom|br)\b', query.lower())
        return int(m.group(1)) if m else None

    def extract_budget(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        for pattern, typ in self.budget_patterns:
            m = re.search(pattern, q)
            if not m:
                continue
            
            if typ == 'range':
                n1 = float(m.group(1)); n2 = float(m.group(2))
                unit = m.group(3) if m.lastindex >= 3 else None
                return {'min_amount': self.parse_price(n1, unit), 'max_amount': self.parse_price(n2, unit), 'type': 'range'}
            else:
                num = float(m.group(1))
                unit = m.group(2) if m.lastindex >= 2 else None
                return {'amount': self.parse_price(num, unit), 'type': typ}
        return {}

    def extract_possession(self, query: str) -> Optional[str]:
        q = query.lower()
        for status, keys in self.possession_map.items():
            if any(k in q for k in keys):
                return status.title()
        return None

    def extract_amenities(self, query: str) -> List[str]:
        found = []
        q = query.lower()
        for amen, keys in self.amenity_keywords.items():
            if any(k in q for k in keys):
                found.append(amen)
        return found

    def parse(self, query: str) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        filters['city'] = self.extract_city(query)
        filters['locality'] = self.extract_locality(query, filters.get('city'))
        bhk = self.extract_bhk(query)
        if bhk: filters['bhk'] = bhk
        budget = self.extract_budget(query)
        filters.update(budget)
        pos = self.extract_possession(query)
        if pos: filters['possession'] = pos
        am = self.extract_amenities(query)
        if am: filters['amenities'] = am
        return {k: v for k, v in filters.items() if v is not None and v != [] and v != ''}

# -----------------------
# Search engine
# -----------------------
class PropertySearchEngine:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        
        # City filter
        if 'city' in filters:
            result = result[result['City'].str.lower() == filters['city'].lower()]

        # BHK filter
        if 'bhk' in filters and 'BHK' in result.columns:
            result = result[result['BHK'].astype(str).str.contains(str(filters['bhk']), na=False)]

        # Price/Budget filter
        if 'type' in filters and 'Price_INR' in result.columns:
            t = filters['type']
            try:
                price_col = result['Price_INR'].astype(float)
                if t == 'under' and 'amount' in filters:
                    result = result[price_col <= float(filters['amount'])]
                elif t == 'above' and 'amount' in filters:
                    result = result[price_col >= float(filters['amount'])]
                elif t == 'range' and 'min_amount' in filters:
                    result = result[(price_col >= float(filters['min_amount'])) & (price_col <= float(filters['max_amount']))]
            except:
                pass

        # Locality filter
        if 'locality' in filters and 'Locality' in result.columns:
            result = result[result['Locality'].str.lower().str.contains(filters['locality'].lower(), na=False)]

        # Possession filter
        if 'possession' in filters and 'Possession' in result.columns:
            result = result[result['Possession'].str.lower().str.contains(filters['possession'].lower(), na=False)]

        # Amenities filter (scoring)
        if 'amenities' in filters and isinstance(filters['amenities'], list) and len(filters['amenities']) > 0 and 'Amenities' in result.columns:
            def score_amenities(amen_list):
                if not isinstance(amen_list, list): return 0
                lower = [str(a).lower() for a in amen_list]
                return sum(1 for req in filters['amenities'] if any(req in a for a in lower))
            
            result['amenity_score'] = result['Amenities'].apply(score_amenities)
            result = result[result['amenity_score'] > 0].sort_values('amenity_score', ascending=False).drop(columns=['amenity_score'])

        return result

    def search_with_fallback(self, filters: Dict[str, Any]) -> tuple:
        """Performs initial search and relaxes filters if no results are found."""
        result = self.apply_filters(self.data, filters)
        relaxed = []

        # Fallback 1: relax BHK
        if len(result) == 0 and 'bhk' in filters:
            relaxed_filters = dict(filters); relaxed_filters.pop('bhk', None)
            temp = self.apply_filters(self.data, relaxed_filters)
            if len(temp) > 0: result = temp; relaxed.append('BHK')

        # Fallback 2: relax locality
        if len(result) == 0 and 'locality' in filters:
            relaxed_filters = {k: v for k, v in filters.items() if k != 'locality'}
            temp = self.apply_filters(self.data, relaxed_filters)
            if len(temp) > 0: result = temp; relaxed.append('locality')

        # Fallback 3: relax possession
        if len(result) == 0 and 'possession' in filters:
            relaxed_filters = {k: v for k, v in filters.items() if k != 'possession'}
            temp = self.apply_filters(self.data, relaxed_filters)
            if len(temp) > 0: result = temp; relaxed.append('possession')

        if len(result) > 0 and 'Price_INR' in result.columns:
            try: result = result.sort_values('Price_INR')
            except Exception: pass

        return result, len(relaxed) > 0, relaxed

# -----------------------
# Summary & card builder
# -----------------------
def generate_summary(df: pd.DataFrame, filters: Dict, fallback: bool, relaxed: List[str]) -> str:
    """Generates a text summary of the search results."""
    if df is None or len(df) == 0:
        city = filters.get('city', 'your area')
        return f"No properties found matching your criteria in {city}. Try adjusting your search parameters like budget, BHK, or locality."
        
    count = len(df)
    city = filters.get('city', 'various cities').title()
    parts = []
    
    if fallback:
        parts.append(f"Found **{count}** projects in **{city}** (relaxed: {', '.join(relaxed)})")
    else:
        parts.append(f"Found **{count}** projects in **{city}**")
        
    if 'Price_INR' in df.columns and df['Price_INR'].notna().any():
        try:
            avg_price = float(df['Price_INR'].astype(float).median())
            parts.append(f"Median price: **{format_price(avg_price)}**")
        except Exception:
            pass
            
    all_amen = []
    if 'Amenities' in df.columns:
        for amen_list in df['Amenities'].dropna():
            if isinstance(amen_list, list):
                all_amen.extend(amen_list)
                
    if all_amen:
        top = [a for a, _ in Counter(all_amen).most_common(3)]
        parts.append(f"Common amenities: {', '.join(top)}")
        
    return ". ".join(parts) + "."

def build_property_cards(df: pd.DataFrame, limit: int = 20) -> List[PropertyCard]:
    """Converts a DataFrame of properties into a list of Pydantic PropertyCard models."""
    cards: List[PropertyCard] = []
    if df is None or len(df) == 0:
        return cards
        
    for _, row in df.head(limit).iterrows():
        amenities = row['Amenities'] if isinstance(row['Amenities'], list) else []
        project_name = str(row.get('Project') or row.get('ProjectName') or "Property")
        slug = re.sub(r'[^a-z0-9-]', '', project_name.lower().replace(' ', '-'))
        city = row.get('City') or "Not Specified"
        locality = row.get('Locality') or "Not Specified"
        bhk_val = row.get('BHK')
        bhk = f"{int(bhk_val)} BHK" if pd.notna(bhk_val) and str(bhk_val).isdigit() else "Varied"

        cards.append(PropertyCard(
            title=project_name,
            city_locality=f"{city} • {locality}",
            bhk=bhk,
            price=format_price(row.get('Price_INR')),
            project_name=project_name,
            possession=str(row.get('Possession') or "Not specified"),
            amenities=[str(a) for a in amenities[:3]],
            cta=f"/project/{slug}"
        ))
    return cards

# -----------------------
# Data loading on startup
# -----------------------
@app.on_event("startup")
async def load_data():
    """Locates, loads, and preprocesses the properties data."""
    global merged_data, cities_list, localities_list
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Pre-defined list of common Indian cities for robust detection
    possible_cities = [
        "Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai", "Hyderabad",
        "Kolkata", "Ahmedabad", "Gurgaon", "Noida", "Jaipur", "Lucknow",
        "Surat", "Indore", "Nagpur", "Bhopal", "Vadodara", "Visakhapatnam"
    ]

    # Find and load the CSV file
    found = None
    possible_paths = [
        os.path.join(base_dir, "data", "merged_properties.csv"),
        os.path.join(base_dir, "..", "data", "merged_properties.csv"),
        os.path.join(base_dir, "merged_properties.csv"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            found = p
            break
            
    if not found:
        print("⚠️ Data file not found. Using sample data.")
        merged_data = pd.DataFrame([{
            'ProjectId': 1, 'Project': 'Sample Property', 'Address': 'Andheri, Mumbai',
            'City': 'Mumbai', 'Locality': 'Andheri', 'BHK': 3,
            'Price_INR': 12000000, 'Possession': 'Ready', 'Amenities': ['Gym', 'Parking']
        }])
    else:
        print(f"✅ Loading data from: {found}")
        try:
            merged_data = pd.read_csv(found)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}. Creating empty data.")
            merged_data = pd.DataFrame()

    if merged_data.empty:
        cities_list = []; localities_list = []
        return

    # --- Data Preprocessing and Normalization ---
    
    # Ensure all required columns exist
    merged_data.rename(columns={'project': 'Project', 'price_inr': 'Price_INR'}, inplace=True)
    if 'Address' not in merged_data.columns:
        merged_data['Address'] = merged_data.apply(lambda r: f"{r.get('Locality','')}, {r.get('City','')}" if 'Locality' in merged_data and 'City' in merged_data else "Not Specified", axis=1)
    
    # Create the 'City' and 'Locality' columns from 'Address' using the list of known cities
    merged_data[['City', 'Locality']] = merged_data['Address'].apply(
        lambda x: pd.Series(parse_address_for_city_locality(x, possible_cities))
    )
    
    # Normalize other columns
    merged_data['Amenities'] = merged_data.get('Amenities', []).apply(normalize_amenities_field)
    merged_data['City'] = merged_data['City'].apply(safe_title)
    merged_data['Locality'] = merged_data['Locality'].apply(safe_title)
    
    # Build the final lists for lookup
    cities_list.extend(sorted(merged_data['City'].dropna().unique().tolist()))
    localities_list.extend(sorted(merged_data['Locality'].dropna().unique().tolist()))
    
    print(f"✅ Loaded records: {len(merged_data)} | Cities found: {len(cities_list)} sample: {cities_list[:6]}")

# -----------------------
# Endpoints
# -----------------------
@app.get("/")
async def root():
    return {
        "message": "NoBrokerage AI Property Search API",
        "status": "running",
        "version": "2.0",
        "properties": len(merged_data) if merged_data is not None else 0,
        "cities": len(cities_list)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "data_loaded": merged_data is not None,
        "total_properties": len(merged_data) if merged_data is not None else 0,
        "cities": len(cities_list),
        "localities": len(localities_list)
    }

@app.get("/cities")
async def get_cities():
    if not cities_list:
        raise HTTPException(status_code=404, detail="No cities found in dataset.")
    return {"cities": cities_list}

@app.get("/localities/{city}")
async def get_localities(city: str):
    if merged_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    df = merged_data[merged_data['City'].str.lower() == city.lower()]
    local = sorted(df['Locality'].dropna().unique().tolist())
    return {"city": city, "localities": [l for l in local if l and l != 'Not Specified']}

@app.get("/properties")
async def list_properties(city: Optional[str] = None, limit: int = 20):
    if merged_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    df = merged_data
    if city:
        df = df[df['City'].str.lower() == city.lower()]
    return {"count": len(df), "properties": df.head(limit).to_dict(orient="records")}

@app.post("/api/query", response_model=QueryResponse)
async def query_properties(request: QueryRequest):
    if merged_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    parser = NaturalLanguageParser(cities_list, localities_list)
    filters = parser.parse(q)

    # Contextual city fallback/prompt
    if 'city' not in filters:
        if request.context and request.context.get('last_city'):
            filters['city'] = request.context.get('last_city')
        else:
            available = ', '.join(cities_list[:10])
            return QueryResponse(
                summary=f"Please specify a city in your query. Available cities: {available}, ...",
                cards=[],
                meta={"appliedFilters": filters, "error": "no_city_specified", "availableCities": cities_list}
            )

    search_engine = PropertySearchEngine(merged_data)
    results, fallback_used, relaxed_filters = search_engine.search_with_fallback(filters)

    summary = generate_summary(results, filters, fallback_used, relaxed_filters)
    cards = build_property_cards(results, limit=20)

    meta = {
        "appliedFilters": filters,
        "fallbackUsed": fallback_used,
        "relaxedFilters": relaxed_filters,
        "totalResults": len(results),
        "query": q
    }
    
    return QueryResponse(summary=summary, cards=cards, meta=meta)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)