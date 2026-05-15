from openai import OpenAI
import numpy as np

client = OpenAI()


photos = [
    {
        "path": "photos/kamakura_001.jpg",
        "text": "rainy evening street near a temple, wet stone path, umbrellas"
    },
    {
        "path": "photos/chiba_road_002.jpg",
        "text": "motorcycle on coastal road, cloudy sky, sea, Japan"
    },
    {
        "path": "photos/tokyo_night_003.jpg",
        "text": "night city street, neon lights, salarymen, reflections"
    },
]


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Build tiny in-memory index
for photo in photos:
    photo["embedding"] = get_embedding(photo["text"])


query = "lonely rainy Japanese temple street"
query_embedding = get_embedding(query)


results = []

for photo in photos:
    score = cosine_similarity(query_embedding, photo["embedding"])
    results.append({
        "path": photo["path"],
        "text": photo["text"],
        "score": score,
    })


results.sort(key=lambda x: x["score"], reverse=True)

for r in results:
    print(round(r["score"], 4), r["path"], "-", r["text"])