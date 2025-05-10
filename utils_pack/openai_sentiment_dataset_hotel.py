#%%
from openai import OpenAI
import pandas as pd
import json
import ast
import os


client=OpenAI(api_key="xxxxxxxxxxxxxx")
#%%

dataset=pd.read_csv("../Data/sentiment_dataset_huggingface_hotel.csv")
print(dataset.head())


dataset_100=dataset.iloc[:100]
dataset_full=dataset.iloc[100:]

#%%
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_review_insights",
            "description": "Extracts key information from a product review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative"]
                    },
                    "key_points": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "suggestions": {
                        "type": "string",
                        "description": "Suggestions for improvement based on the review."
                    }
                },
                "required": ["sentiment", "key_points"]
            }
        }
    }
]

def analizar_review(review):

    try:
        response=client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "user", "content": review}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_review_insights"}}
        )

        tool_call=response.choices[0].message.tool_calls[0]
        return tool_call.function.arguments
    except Exception as e:
        print(f"Error al analizar la reseña: {e}")
        return None

#dataset_100["insights"]=dataset_100["review"].apply(analizar_review)
#dataset_full["insights"]=dataset_full["review"].apply(analizar_review)

#%%

dataset_100_copy=dataset_100.copy()
dataset_full_copy=dataset_full.copy()


#print(type(dataset_100_copy["insights"]))


# %%

dataset=pd.merge(dataset_100_copy, dataset_full_copy, how="outer")

dataset["insights_conv"] = dataset["insights"].apply(lambda x: ast.literal_eval(x) if x else None)
# Extract key points and suggestions into new columns
dataset["key_points"] = dataset["insights_conv"].apply(
    lambda x: x.get("key_points", []) if x else []
)
dataset["suggestions"] = dataset["insights_conv"].apply(
    lambda x: x.get("suggestions", "No suggestion") if x else "No suggestion"
)

# Print the first 3 rows to verify
for i in range(3):
    key_points = dataset["key_points"][i]
    suggestion = dataset["suggestions"][i]
    print(f"Row {i} - Key Points: {key_points}, Suggestion: {suggestion}")
# %%

print(dataset.info())
os.makedirs("../Data", exist_ok=True)
dataset.to_csv("../Data/sentiment_dataset_openai_hotel.csv", index=False)

print("✅ Datos generados correctamente.")
