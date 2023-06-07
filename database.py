import pymongo
from bson import ObjectId

URL = "mongodb://streamlit:8z1jvVFHVuH3pipFJOUX87sNa10nSKIpSYyg2H1FHqRKVNGyvO4HQoaidQEyKj3zK87NWqsX7YZaACDbzZT85w==@streamlit.mongo.cosmos.azure.com:10255/?ssl=true&retrywrites=false&replicaSet=globaldb&maxIdleTimeMS=120000&appName=@streamlit@"


class FilmsDB:
    def __init__(self, mongo_url=URL) -> None:
        self.client = pymongo.MongoClient(mongo_url)
        self.db = self.client["filmsClient"]
        self.collection = self.db["films"]

    def get_all_keywords(self):
        return self.collection.find({}, {'_id': 1, 'keywords': 1})

    def get_by_ids(self, indexes):
        id_to_index = {ObjectId(x): i for i, x in enumerate(indexes)}
        documents = self.collection.find({"_id": {"$in": list(map(ObjectId, indexes))}})

        # Sort the documents based on the original order of IDs
        sorted_documents = sorted(documents, key=lambda doc: id_to_index[doc["_id"]])

        return sorted_documents

    def count_all(self):
        return self.collection.count_documents({})
