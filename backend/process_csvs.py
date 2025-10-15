import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import List, Dict, Optional

class NoBrokerageDataProcessor:
    """
    Process and merge NoBrokerage CSV files with exact column names.
    Handles: Project.csv, ProjectAddress.csv, ProjectConfiguration.csv, ProjectConfigurationVariant.csv
    """
    
    def __init__(self, data_dir='./data'):
        self.data_dir = Path(data_dir)
        self.project_df = None
        self.address_df = None
        self.config_df = None
        self.variant_df = None
        self.merged_data = None
        
    def load_csvs(self):
        """Load all 4 CSV files"""
        print("="*70)
        print(" "*20 + "LOADING CSV FILES")
        print("="*70)
        
        try:
            self.project_df = pd.read_csv(self.data_dir / 'Project.csv')
            self.project_df.columns = self.project_df.columns.str.strip()
            print(f"✓ Loaded Project.csv: {len(self.project_df)} rows, {len(self.project_df.columns)} columns")
            print(f"  Columns: {', '.join(self.project_df.columns.tolist())}")
        except Exception as e:
            print(f"✗ Error loading Project.csv: {e}")
            return False
        
        try:
            self.address_df = pd.read_csv(self.data_dir / 'ProjectAddress.csv')
            self.address_df.columns = self.address_df.columns.str.strip()
            print(f"✓ Loaded ProjectAddress.csv: {len(self.address_df)} rows, {len(self.address_df.columns)} columns")
            print(f"  Columns: {', '.join(self.address_df.columns.tolist())}")
        except Exception as e:
            print(f"✗ Error loading ProjectAddress.csv: {e}")
            return False
        
        try:
            self.config_df = pd.read_csv(self.data_dir / 'ProjectConfiguration.csv')
            self.config_df.columns = self.config_df.columns.str.strip()
            print(f"✓ Loaded ProjectConfiguration.csv: {len(self.config_df)} rows, {len(self.config_df.columns)} columns")
            print(f"  Columns: {', '.join(self.config_df.columns.tolist())}")
        except Exception as e:
            print(f"✗ Error loading ProjectConfiguration.csv: {e}")
            return False
        
        try:
            self.variant_df = pd.read_csv(self.data_dir / 'ProjectConfigurationVariant.csv')
            self.variant_df.columns = self.variant_df.columns.str.strip()
            print(f"✓ Loaded ProjectConfigurationVariant.csv: {len(self.variant_df)} rows, {len(self.variant_df.columns)} columns")
            print(f"  Columns: {', '.join(self.variant_df.columns.tolist())}")
        except Exception as e:
            print(f"✗ Error loading ProjectConfigurationVariant.csv: {e}")
            return False
        
        return True
    
    def parse_price_to_inr(self, price_str):
        """Convert price strings to numeric INR"""
        if pd.isna(price_str) or price_str == '':
            return None
        
        # If already numeric, return as is
        if isinstance(price_str, (int, float)):
            return float(price_str)
        
        price_str = str(price_str).lower().replace('₹', '').replace(',', '').strip()
        
        # Handle ranges - take minimum
        if '-' in price_str:
            price_str = price_str.split('-')[0].strip()
        
        # Extract number and unit
        match = re.search(r'([\d\.]+)\s*(cr|crore|l|lac|lakh|k|thousand|million)?', price_str)
        if match:
            num = float(match.group(1))
            unit = match.group(2)
            
            if unit in ['cr', 'crore']:
                return num * 10000000
            elif unit in ['l', 'lac', 'lakh']:
                return num * 100000
            elif unit in ['k', 'thousand']:
                return num * 1000
            elif unit == 'million':
                return num * 1000000
            else:
                # No unit - check magnitude
                if num > 10000000:  # Already in INR
                    return num
                elif num > 100000:  # Likely in lakhs or above
                    return num
                elif num > 1000:  # Likely in thousands
                    return num * 1000
                elif num > 10:  # Likely in lakhs
                    return num * 100000
                else:  # Likely in crores
                    return num * 10000000
        
        return None
    
    def extract_bhk(self, bhk_str):
        """Extract BHK number from customBHK or type field"""
        if pd.isna(bhk_str):
            return None
        
        bhk_str = str(bhk_str).lower()
        
        # Try multiple patterns
        patterns = [
            r'(\d+)\s*bhk',
            r'(\d+)\s*bed',
            r'(\d+)\s*br',
            r'(\d+)bhk',
            r'(\d+)\s*b',
            r'^(\d+)$'  # Just a number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, bhk_str)
            if match:
                return int(match.group(1))
        
        return None
    
    def normalize_possession(self, poss_date):
        """Normalize possession date to status"""
        if pd.isna(poss_date):
            return "Not Specified"
        
        poss_str = str(poss_date).lower()
        
        # Check for ready/immediate keywords
        ready_keywords = ['ready', 'immediate', 'available', 'possession']
        if any(kw in poss_str for kw in ready_keywords):
            return "Ready"
        
        # Check for under construction keywords
        uc_keywords = ['under construction', 'uc', 'upcoming', 'construction']
        if any(kw in poss_str for kw in uc_keywords):
            return "Under Construction"
        
        # Try to parse date
        try:
            from datetime import datetime
            # Try different date formats
            date_formats = ['%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%d/%m/%Y']
            for fmt in date_formats:
                try:
                    poss_date_obj = datetime.strptime(poss_str.split()[0], fmt)
                    today = datetime.now()
                    
                    if poss_date_obj <= today:
                        return "Ready"
                    else:
                        return "Under Construction"
                except:
                    continue
        except:
            pass
        
        return "Not Specified"
    
    def get_city_name(self, city_id):
        """Map cityId to city name (you can expand this mapping)"""
        # This is a placeholder - you should populate with actual city mappings
        city_map = {
            1: 'Mumbai',
            2: 'Pune',
            3: 'Bangalore',
            4: 'Delhi',
            5: 'Hyderabad',
            6: 'Chennai',
            7: 'Kolkata',
            8: 'Ahmedabad',
        }
        
        if pd.isna(city_id):
            return "Not Specified"
        
        try:
            city_id = int(city_id)
            return city_map.get(city_id, f"City_{city_id}")
        except:
            return "Not Specified"
    
    def get_locality_name(self, locality_id):
        """Map localityId to locality name (you can expand this mapping)"""
        # Placeholder - you should populate with actual locality mappings
        locality_map = {
            1: 'Andheri',
            2: 'Bandra',
            3: 'Wakad',
            4: 'Baner',
            5: 'Whitefield',
            6: 'Electronic City',
        }
        
        if pd.isna(locality_id):
            return "Not Specified"
        
        try:
            locality_id = int(locality_id)
            return locality_map.get(locality_id, f"Locality_{locality_id}")
        except:
            return "Not Specified"
    
    def merge_all(self):
        """Merge all 4 dataframes into one"""
        print("\n" + "="*70)
        print(" "*25 + "MERGING FILES")
        print("="*70)
        
        # Debug: Show data before merging
        print(f"\nInitial data counts:")
        print(f"   Project.csv: {len(self.project_df)} rows")
        print(f"   ProjectAddress.csv: {len(self.address_df)} rows")
        print(f"   ProjectConfigurations.csv: {len(self.config_df)} rows")
        print(f"   ProjectConfigurationsVariant.csv: {len(self.variant_df)} rows")
        
        # Check column names
        print(f"\nChecking key columns:")
        print(f"   Project.csv 'id' column: {self.project_df['id'].head(3).tolist()}")
        print(f"   ProjectAddress.csv 'projectId' column: {self.address_df['projectId'].head(3).tolist() if 'projectId' in self.address_df.columns else 'NOT FOUND'}")
        print(f"   ProjectConfigurations.csv 'projectId' column: {self.config_df['projectId'].head(3).tolist() if 'projectId' in self.config_df.columns else 'NOT FOUND'}")
        
        # Step 1: Merge Project with Address (on projectId)
        print("\n1. Merging Project.csv with ProjectAddress.csv...")
        if 'projectId' in self.address_df.columns:
            merged = self.project_df.merge(
                self.address_df,
                left_on='id',
                right_on='projectId',
                how='left',
                suffixes=('', '_address')
            )
            print(f"   ✓ Result: {len(merged)} rows")
        else:
            print("   ⚠ 'projectId' not found in ProjectAddress.csv, skipping address merge")
            merged = self.project_df.copy()
        
        # Step 2: Merge with Configurations (on projectId)
        print("\n2. Merging with ProjectConfigurations.csv...")
        if 'projectId' in self.config_df.columns:
            merged = merged.merge(
                self.config_df,
                left_on='id',
                right_on='projectId',
                how='left',
                suffixes=('', '_config')
            )
            print(f"   ✓ Result: {len(merged)} rows")
            
            # Debug: check if we have configuration IDs
            if 'id_config' in merged.columns:
                print(f"   Configuration IDs: {merged['id_config'].head(3).tolist()}")
        else:
            print("   ⚠ 'projectId' not found in ProjectConfiguration.csv, skipping config merge")
        
        # Step 3: Merge with Variant (on configurationId)
        print("\n3. Merging with ProjectConfigurationVariant.csv...")
        
        # Find the configuration ID column in merged data
        config_id_col = None
        for col in ['id_config', 'configId', 'configuration_id']:
            if col in merged.columns:
                config_id_col = col
                break
        
        if config_id_col and 'configurationId' in self.variant_df.columns:
            print(f"   Using '{config_id_col}' to join with 'configurationId'")
            merged = merged.merge(
                self.variant_df,
                left_on=config_id_col,
                right_on='configurationId',
                how='left',
                suffixes=('', '_variant')
            )
            print(f"   ✓ Result: {len(merged)} rows")
        else:
            print(f"   ⚠ Could not find configuration ID column. Available: {merged.columns.tolist()[:10]}")
            print("   Skipping variant merge")
        
        print(f"\n   ✓ Final merged dataframe: {len(merged)} rows, {len(merged.columns)} columns")
        
        # Show sample of merged data
        if len(merged) > 0:
            print(f"\n   Sample merged data:")
            print(merged.head(2)[['id', 'projectName', 'cityId', 'localityId']].to_string() if all(c in merged.columns for c in ['id', 'projectName', 'cityId', 'localityId']) else "")
        
        return merged
    
    def create_standardized_output(self, merged_df):
        """Create standardized output with clean column names"""
        print("\n" + "="*70)
        print(" "*20 + "CREATING STANDARD OUTPUT")
        print("="*70)
        
        if len(merged_df) == 0:
            print("✗ ERROR: Merged dataframe is empty!")
            print("Please check if your CSV files have data and the ID columns match correctly.")
            return pd.DataFrame()
        
        print(f"\nProcessing {len(merged_df)} rows...")
        
        output = pd.DataFrame()
        
        # Helper function to safely get column
        def safe_get_col(df, col_name, default=None):
            if col_name in df.columns:
                return df[col_name]
            else:
                print(f"   ⚠ Column '{col_name}' not found, using default")
                return pd.Series([default] * len(df))
        
        # Project ID
        output['ProjectId'] = safe_get_col(merged_df, 'id')
        
        # Project Name
        output['Project'] = safe_get_col(merged_df, 'projectName', 'Unnamed Project')
        
        # Project Type & Category
        output['ProjectType'] = safe_get_col(merged_df, 'projectType')
        output['ProjectCategory'] = safe_get_col(merged_df, 'projectCategory')
        
        # City - map from cityId
        print("\nMapping cities...")
        city_series = safe_get_col(merged_df, 'cityId')
        output['City'] = city_series.apply(self.get_city_name)
        unique_cities = output['City'].unique()
        print(f"✓ Found cities: {', '.join([c for c in unique_cities if c != 'Not Specified'][:10])}")
        
        # Locality - map from localityId
        print("\nMapping localities...")
        locality_series = safe_get_col(merged_df, 'localityId')
        output['Locality'] = locality_series.apply(self.get_locality_name)
        print(f"✓ Found {output['Locality'].nunique()} unique localities")
        
        # Sub Locality
        output['SubLocality'] = safe_get_col(merged_df, 'subLocalityId')
        
        # Address
        output['Address'] = safe_get_col(merged_df, 'fullAddress', '')
        output['Landmark'] = safe_get_col(merged_df, 'landmark', '')
        output['Pincode'] = safe_get_col(merged_df, 'pincode', '')
        
        # Configuration/BHK
        print("\nExtracting BHK information...")
        if 'customBHK' in merged_df.columns:
            output['Configuration'] = merged_df['customBHK']
            output['BHK'] = merged_df['customBHK'].apply(self.extract_bhk)
        elif 'type' in merged_df.columns:
            output['Configuration'] = merged_df['type']
            output['BHK'] = merged_df['type'].apply(self.extract_bhk)
        else:
            output['Configuration'] = 'Not Specified'
            output['BHK'] = None
            print("   ⚠ No BHK/configuration column found")
        
        bhk_dist = output['BHK'].value_counts().sort_index()
        if len(bhk_dist) > 0:
            print(f"✓ BHK distribution: {dict(bhk_dist.head(10))}")
        
        # Property Category & Type
        output['PropertyCategory'] = safe_get_col(merged_df, 'propertyCategory')
        output['PropertyType'] = safe_get_col(merged_df, 'type')
        
        # Price
        print("\nProcessing prices...")
        if 'price' in merged_df.columns:
            output['PriceRaw'] = merged_df['price']
            output['Price_INR'] = merged_df['price'].apply(self.parse_price_to_inr)
            
            valid_prices = output['Price_INR'].dropna()
            if len(valid_prices) > 0:
                print(f"✓ Price range: ₹{valid_prices.min()/100000:.2f}L - ₹{valid_prices.max()/10000000:.2f}Cr")
                print(f"✓ Average price: ₹{valid_prices.mean()/100000:.2f}L")
            else:
                print("   ⚠ No valid prices found")
        else:
            output['PriceRaw'] = None
            output['Price_INR'] = None
            print("   ⚠ No price column found")
        
        # Area
        output['CarpetArea'] = safe_get_col(merged_df, 'carpetArea')
        
        # Possession
        print("\nProcessing possession status...")
        poss_series = safe_get_col(merged_df, 'possessionDate')
        output['Possession'] = poss_series.apply(self.normalize_possession)
        poss_dist = output['Possession'].value_counts()
        print(f"✓ Possession: {dict(poss_dist)}")
        
        # Property Details
        output['Bathrooms'] = safe_get_col(merged_df, 'bathrooms')
        output['Balcony'] = safe_get_col(merged_df, 'balcony')
        output['ParkingType'] = safe_get_col(merged_df, 'parkingType')
        output['FurnishedType'] = safe_get_col(merged_df, 'furnishedType')
        output['Lift'] = safe_get_col(merged_df, 'lift')
        output['AgeOfProperty'] = safe_get_col(merged_df, 'ageOfProperty')
        
        # Amenities - construct from available facilities
        print("\nConstructing amenities...")
        amenities_list = []
        for idx, row in merged_df.iterrows():
            amen = []
            
            if pd.notna(row.get('lift')) and str(row.get('lift')).lower() in ['yes', 'true', '1']:
                amen.append('Lift')
            if pd.notna(row.get('parkingType')):
                amen.append('Parking')
            if pd.notna(row.get('furnishedType')) and str(row.get('furnishedType')).lower() not in ['unfurnished', 'none', '']:
                amen.append('Furnished')
            if pd.notna(row.get('balcony')) and str(row.get('balcony')).lower() in ['yes', 'true', '1'] or (isinstance(row.get('balcony'), (int, float)) and row.get('balcony') > 0):
                amen.append('Balcony')
            
            amenities_list.append(amen)
        
        output['Amenities'] = amenities_list
        
        # Additional Info
        output['ProjectSummary'] = safe_get_col(merged_df, 'projectSummary', '')
        output['AboutProperty'] = safe_get_col(merged_df, 'aboutProperty', '')
        output['MaintenanceCharges'] = safe_get_col(merged_df, 'maintenanceCharges')
        
        # Images
        output['FloorPlanImage'] = safe_get_col(merged_df, 'floorPlanImage', '')
        output['PropertyImages'] = safe_get_col(merged_df, 'propertyImages', '')
        
        # Status & Dates
        output['Status'] = safe_get_col(merged_df, 'status')
        output['ProjectAge'] = safe_get_col(merged_df, 'projectAge')
        output['ReraId'] = safe_get_col(merged_df, 'reraId', '')
        output['Slug'] = safe_get_col(merged_df, 'slug', '')
        
        # Clean up - remove rows with no city or project name
        print("\nCleaning data...")
        initial_count = len(output)
        
        # Don't filter out "Not Specified" cities for now - keep all data
        output = output.dropna(subset=['Project'])
        output = output[output['Project'] != 'Unnamed Project']
        
        # Remove duplicates
        if len(output) > 0:
            output = output.drop_duplicates(subset=['ProjectId', 'Configuration'], keep='first')
        
        print(f"✓ Removed {initial_count - len(output)} invalid/duplicate rows")
        print(f"✓ Final output: {len(output)} rows")
        
        if len(output) == 0:
            print("\n✗ WARNING: Final output is empty!")
            print("   This usually means:")
            print("   1. CSV files don't have matching IDs between them")
            print("   2. All projects were filtered out due to missing data")
            print("   3. Column names don't match expected names")
        
        return output
    
    def print_statistics(self):
        """Print detailed statistics"""
        if self.merged_data is None:
            return
        
        df = self.merged_data
        
        print("\n" + "="*70)
        print(" "*25 + "DATA STATISTICS")
        print("="*70)
        
        print(f"\n📊 Overview:")
        print(f"   Total Properties: {len(df)}")
        print(f"   Unique Projects: {df['ProjectId'].nunique()}")
        print(f"   Cities: {df['City'].nunique()}")
        print(f"   Localities: {df['Locality'].nunique()}")
        
        print(f"\n🏙️  Top 10 Cities:")
        city_counts = df['City'].value_counts().head(10)
        for city, count in city_counts.items():
            print(f"   {city}: {count}")
        
        print(f"\n📍 Top 10 Localities:")
        loc_counts = df['Locality'].value_counts().head(10)
        for loc, count in loc_counts.items():
            print(f"   {loc}: {count}")
        
        print(f"\n🛏️  BHK Distribution:")
        bhk_dist = df['BHK'].value_counts().sort_index()
        for bhk, count in bhk_dist.items():
            if pd.notna(bhk):
                print(f"   {int(bhk)} BHK: {count}")
        
        print(f"\n🏗️  Possession Status:")
        poss_counts = df['Possession'].value_counts()
        for status, count in poss_counts.items():
            print(f"   {status}: {count}")
        
        if df['Price_INR'].notna().any():
            print(f"\n💰 Price Statistics:")
            print(f"   Min: ₹{df['Price_INR'].min()/100000:.2f} L")
            print(f"   Max: ₹{df['Price_INR'].max()/10000000:.2f} Cr")
            print(f"   Mean: ₹{df['Price_INR'].mean()/100000:.2f} L")
            print(f"   Median: ₹{df['Price_INR'].median()/100000:.2f} L")
        
        if 'ProjectType' in df.columns:
            print(f"\n🏘️  Project Types:")
            type_counts = df['ProjectType'].value_counts().head(5)
            for ptype, count in type_counts.items():
                print(f"   {ptype}: {count}")
    
    def save_merged_data(self, output_file='merged_properties.csv'):
        """Save merged data to CSV and JSON"""
        if self.merged_data is None:
            print("✗ No data to save")
            return
        
        print("\n" + "="*70)
        print(" "*25 + "SAVING OUTPUT")
        print("="*70)
        
        output_path = self.data_dir / output_file
        
        # Convert amenities list to string for CSV
        save_df = self.merged_data.copy()
        save_df['Amenities'] = save_df['Amenities'].apply(
            lambda x: ','.join(x) if isinstance(x, list) else ''
        )
        
        # Save CSV
        save_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved CSV: {output_path}")
        print(f"  Rows: {len(save_df)}")
        print(f"  Columns: {len(save_df.columns)}")
        
        # Save JSON
        json_path = self.data_dir / output_file.replace('.csv', '.json')
        save_df.head(100).to_json(json_path, orient='records', indent=2)  # Save only first 100 for JSON
        print(f"✓ Saved JSON (sample): {json_path}")
        
        # Save column info
        info_path = self.data_dir / 'column_info.txt'
        with open(info_path, 'w') as f:
            f.write("Column Names and Types:\n")
            f.write("="*50 + "\n\n")
            for col in save_df.columns:
                dtype = save_df[col].dtype
                non_null = save_df[col].notna().sum()
                f.write(f"{col}: {dtype} ({non_null}/{len(save_df)} non-null)\n")
        print(f"✓ Saved column info: {info_path}")
    
    def process(self):
        """Main processing pipeline"""
        print("\n" + "="*70)
        print(" "*15 + "NOBROKERAGE DATA PROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Load CSVs
        if not self.load_csvs():
            print("\n✗ Failed to load CSV files")
            return None
        
        # Step 2: Merge all files
        merged = self.merge_all()
        
        if merged is None or len(merged) == 0:
            print("\n✗ Merge failed or resulted in empty dataframe")
            return None
        
        # Step 3: Create standardized output
        self.merged_data = self.create_standardized_output(merged)
        
        # Step 4: Print statistics
        self.print_statistics()
        
        # Step 5: Save output
        self.save_merged_data()
        
        print("\n" + "="*70)
        print(" "*25 + "PROCESSING COMPLETE!")
        print("="*70)
        print("\n✅ Next steps:")
        print("   1. Review 'data/merged_properties.csv'")
        print("   2. Update city/locality mappings if needed")
        print("   3. Start backend: cd backend && uvicorn main:app --reload")
        print("   4. Start frontend: cd frontend && streamlit run app.py")
        
        return self.merged_data

def main():
    """Main execution"""
    processor = NoBrokerageDataProcessor(data_dir='./data')
    result = processor.process()
    
    if result is not None:
        print(f"\n🎉 Successfully processed {len(result)} properties!")
    else:
        print("\n❌ Processing failed. Please check the errors above.")

if __name__ == "__main__":
    main()