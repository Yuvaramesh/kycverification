import requests

API_KEY = "086d202081683926a91efb20a441a132"

# All datasets you want to check
DATASETS = [
    "un_sc_sanctions",
    "us_ofac_sdn",
    "eu_fsf",
    "au_dfat_sanctions",
    "cn_sanctions",
]

BASE_URL = "https://api.opensanctions.org/match"


def match_user(name, country=None, dob=None):
    results = []

    for dataset in DATASETS:
        url = f"{BASE_URL}/{dataset}?api_key={API_KEY}"

        payload = {
            "queries": {"q1": {"schema": "Person", "properties": {"name": [name]}}}
        }

        # Optional enrichment
        if country:
            payload["queries"]["q1"]["properties"]["country"] = [country]

        if dob:
            payload["queries"]["q1"]["properties"]["birthDate"] = [dob]

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()

            if "responses" in data and "q1" in data["responses"]:
                matches = data["responses"]["q1"].get("results", [])

                for match in matches:
                    score = match.get("score", 0)

                    # Filter strong matches
                    if score > 0.7:
                        results.append(
                            {
                                "dataset": dataset,
                                "name": match["entity"].get("caption"),
                                "score": score,
                                "id": match["entity"].get("id"),
                            }
                        )

        else:
            print(f"Error in dataset {dataset}: {response.status_code}")

    return results


# 🔍 Example usage
user_name = "Abu Lulu"
user_country = "Sudan"

matches = match_user(user_name, user_country)

if matches:
    print("⚠️ Match Found:")
    for m in matches:
        print(m)
else:
    print("✅ No match found")
