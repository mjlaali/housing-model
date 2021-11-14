import json
import tempfile

from housing_model.preprocessing.features import get_features, build_feature_template

a_sample = """
{
  "ml_num": "N2447318",
  "hash_id": "4G132z4CzoNa38Bx",
  "id_listing": "N0A9X3jjZNE3vgxV",
  "seo_suffix": "128-Larratt-Lane-Richmond-Hill-L4C0E6-N2447318",
  "photo_url": "https://rendertron.fangintel.com/Live/photos/FULL/1/318/N2447318.jpg?3b6eef4e",
  "house_type_name": "Detached",
  "house_type": "D.",
  "house_style": "2-Storey",
  "address": "128 Larratt Lane",
  "list_status": {
    "public": 1,
    "live": 0,
    "sold": 1,
    "s_r": "Sale",
    "text": "Sold",
    "status": "SLD",
    "feature": null,
    "precon": null,
    "premium": 0,
    "archive": 1
  },
  "community_name": "Westbrook",
  "municipality_name": "Richmond Hill",
  "board": "treb",
  "bedroom": 4,
  "bedroom_plus": 1,
  "bedroom_string": "4+1",
  "washroom": 4,
  "parking": {
    "total": 2,
    "garage_type": "Built-In",
    "garage": 2,
    "parking_type": null,
    "parking": 4,
    "text": "Built-In 2 garage,  4 parking"
  },
  "price": "739,900",
  "price_int": 739900,
  "price_abbr": "0.74M",
  "price_sold": "739,900",
  "price_origin": null,
  "price_sold_int": 739900,
  "price_change_yearly": null,
  "price_change_yearly_text": "Sold for 0% than last bought",
  "map": {
    "lat": 43.8934482,
    "lon": -79.4628076
  },
  "house_area": {
    "area": null,
    "area_note": null,
    "unit": "feet²",
    "estimate": null
  },
  "land": {
    "front": 39,
    "depth": 131,
    "unit": "feet"
  },
  "date_added": "2018-07-23",
  "date_added_days": 527,
  "date_start": "2012-08-24",
  "date_start_days": 2686,
  "date_start_month": 88,
  "date_update": "2014-09-04",
  "date_end": "2012-09-02",
  "date_end_days": 2677,
  "list_days": 9,
  "feature_header": null,
  "analytics": {
    "rent_yield": null,
    "rent_base": "2645074",
    "estimate_price_age": 519,
    "estimate_price_date": "2018-07-31 00:00:00",
    "estimate_price": "1,221,247"
  },
  "scores": {
    "school": 5,
    "land": 7,
    "rent": 0,
    "growth": 7
  },
  "tags": [
    "Sold"
  ],
  "text": {
    "list_date_short": "Sold 2012-09-02",
    "list_date_long": "Sold 2012-09-02, 9 days on market",
    "dom_short": "Sold in Sep,2012",
    "dom_long": "Sold 2677 days ago, 9 days on market",
    "rooms_short": "4+1 bd 4 ba 2 gr",
    "rooms_long": "4+1 Bedroom, 4 Bathroom, 2 Garage"
  },
  "open_house_date": null,
  "sold_month": "2012-09"
}
"""


def test_get_features():
    with tempfile.TemporaryDirectory() as tmp_dir:
        feature_template = build_feature_template(tmp_dir)
        a_listing = json.loads(a_sample)
        features = get_features(a_listing, feature_template)
        assert len(features) == 23
        assert "ml_num" in features
        assert features["ml_num"] == [a_listing["ml_num"]]
