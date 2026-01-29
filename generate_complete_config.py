"""
Generate complete questions_config.json from ALL training data columns
We'll include ALL indicators regardless of whether they have trained models
"""
import pandas as pd
import json
from datetime import datetime

# Load training data to get all indicators
df = pd.read_csv('data/training/Assessment_AI_Training_Data.csv')

print(f"Total columns in dataset: {len(df.columns)}")
print()

# Define metadata columns to exclude
metadata_cols = {
    'plan_id', 'assessment_name', 'user_id', 'user_name', 'workspace_id', 
    'workspace_name', 'email_id', 'hired_workers', 'Unique identifier', 
    'address', 'county', 'plan_year', 'country_code', 'locale', 'state', 
    'zipcode', 'longitude', 'latitude', 'crop_name', 'area', 
    'additional_farm_area', 'irrigation', 'Partner', 'sub_partner', 
    'plan_created', 'plan_modified', 'plan_deleted', 'Submitted', 
    'unit_type', 'yield_unit', 'nitrogen_unit', 'essential_percentage',
    'intermediate_percentage', 'advanced_percentage', 'sai_score'
}

# Get indicator columns
indicator_cols = [col for col in df.columns 
                  if col not in metadata_cols 
                  and not col.startswith('Carbon-score')
                  and col != 'Reason for failure of Carbon-score generation']

print(f"Found {len(indicator_cols)} indicator columns")
print()

# Group by category prefix
categories = {}
for col in indicator_cols:
    # Extract category code (e.g., "BH" from "BH-1")
    if '-' in col or '.' in col:
        cat_code = col.split('-')[0].split('.')[0]
    else:
        cat_code = "OTHER"
    
    if cat_code not in categories:
        categories[cat_code] = []
    categories[cat_code].append(col)

# Create questions config structure
config = {
    "version": "2.0",
    "lastUpdated": datetime.now().strftime("%Y-%m-%d"),
    "totalIndicators": len(indicator_cols),
    "note": "Auto-generated from training data columns",
    "categories": []
}

# Category names mapping
category_names = {
    "BH": "Biodiversity and Habitat",
    "BP": "Biodiversity - Pollinators",
    "CE": "Community Engagement",
    "CW": "Community Welfare",
    "CT": "Cultural Traditions",
    "CC": "Community Collaboration",
    "HS": "Health and Safety - Workers",
    "HW": "Health and Safety - Worksite",
    "HP": "Health and Safety - Protection",
    "HA": "Health and Safety - Animal",
    "HH": "Health and Safety - Human Health",
    "OR": "Operational Records",
    "OP": "Operational Planning",
    "OS": "Operational Systems",
    "OT": "Operational Technology",
    "OF": "Operational Fuel",
    "OE": "Operational Electricity",
    "OM": "Operational Machinery",
    "OW": "Operational Waste",
    "SR": "Soil - Runoff",
    "SW": "Soil - Wind",
    "SF": "Soil - Field Management",
    "SC": "Soil - Crop Rotation",
    "SM": "Soil - Monitoring",
    "WI": "Water - Infrastructure",
    "WP": "Water - Planning",
    "WQ": "Water - Quality",
    "CM": "Community Management",
    "CO": "Conservation",
    "FM": "Farm Management",
    "IO": "Irrigation Operations",
    "IM": "Irrigation Management",
    "LM": "Land Management",
    "NM": "Nutrient Management",
    "PM": "Pest Management",
    "RM": "Regulatory Management",
    "ST": "Sustainability",
    "7": "Additional Standards",
    "8": "Additional Requirements",
    "OTHER": "Other Indicators"
}

for cat_code in sorted(categories.keys()):
    indicators = sorted(categories[cat_code])
    
    # Check how much data each indicator has
    data_coverage = {}
    for indicator in indicators:
        non_null = df[indicator].notna().sum()
        total = len(df)
        coverage_pct = (non_null / total) * 100
        data_coverage[indicator] = {
            'samples': non_null,
            'coverage': coverage_pct
        }
    
    category = {
        "categoryId": cat_code,
        "categoryName": category_names.get(cat_code, f"Category {cat_code}"),
        "questionCount": len(indicators),
        "questions": []
    }
    
    for indicator in indicators:
        # Get sample values from data (limit to 5 unique values)
        sample_values = df[indicator].dropna().unique()[:5].tolist()
        coverage_info = data_coverage[indicator]
        
        question = {
            "indicatorCode": indicator,
            "description": f"Assessment question for {indicator}",
            "type": "text",
            "required": False,
            "dataCoverage": f"{coverage_info['coverage']:.1f}%",
            "trainingSamples": int(coverage_info['samples']),  # Convert to Python int
            "sampleValues": [str(v)[:100] for v in sample_values if v and str(v) != 'nan'][:3]
        }
        category["questions"].append(question)
    
    config["categories"].append(category)

# Save to file
output_file = 'config/questions_config_FULL.json'
with open(output_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Generated complete config with {config['totalIndicators']} indicators")
print(f"📁 Saved to: {output_file}")
print()
print("Summary by category:")
total_questions = 0
for cat in config['categories']:
    print(f"   {cat['categoryId']:10s} - {cat['categoryName']:50s} ({cat['questionCount']:3d} indicators)")
    total_questions += cat['questionCount']

print()
print(f"📊 TOTAL: {total_questions} indicators in config")
print()
print("Next steps:")
print("1. Review the generated file: config/questions_config_FULL.json")
print("2. Replace the old config: mv config/questions_config_FULL.json config/questions_config.json")
print("3. Your AI will now be able to predict ALL indicators!")
