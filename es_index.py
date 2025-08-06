from pymongo import MongoClient
from elasticsearch import Elasticsearch, helpers
from datetime import datetime
from bson.objectid import ObjectId
import json
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(filename='es_index.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


MONGO_URI = "mongodb://abcd:123456@192.168.72.233:27000/?authMechanism=DEFAULT"
DB_NAME = "crawledData"
ES_INDEX = "corpus_collection_index"
ES_HOST = "http://localhost:9201"

BATCH_SIZE = 1000

# Extract and normalize a document
def normalize(doc, collection_name):
    def safe_date(value):
        if isinstance(value, dict) and "$date" in value:
            return value["$date"]
        elif isinstance(value, str):
            return value  # Might still be valid
        else:
            return datetime.now().isoformat()

    try:
        return {
            "_index": ES_INDEX,
            "_id": str(doc["_id"]["$oid"]) if isinstance(doc["_id"], dict) and "$oid" in doc["_id"] else str(doc["_id"]),
            "_source": {
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "source": doc.get("source", ""),
                "source_type": doc.get("source_type", ""),
                "publish_date": safe_date(doc.get("publish_date")),
                "parse_time": safe_date(doc.get("parse_time")),
                "categories": doc.get("categories", ""),
                "domain": doc.get("domain", ""),
                "is_bangla": bool(doc.get("is_bangla", False)),
                "word_count": int(doc.get("word_count", 0)),
                "sentence_count": int(doc.get("sentence_count", 0)),
                "_collection": collection_name,
            }
        }
    except Exception as e:
        logging.error(f"Normalization error: {e}, doc: {doc}")
        raise e



def index_to_elasticsearch():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    es = Elasticsearch(ES_HOST)

    # collections = db.list_collection_names()
    # collections.sort()
    # collections.reverse()

    # with open("indexes.json", "r") as f:
    #     data = json.load(f)

    # collections = list(data["collection"])
    # collections.sort()
    # collections.reverse()

    collections = ["bd-pratidin.com"]

    # Initialize report dictionary
    report_dict = {
        "total_collections": len(collections),
        "collection": {
            col: {
                "total_docs": 0,
                "indexed_docs": 0,
                "errors": 0
            } for col in collections
        }
    }
    with client.start_session() as session:
        for col in collections:
            try:
                total_docs = db[col].count_documents({}, session=session)
                report_dict["collection"][col]["total_docs"] = total_docs

                cursor = db[col].find({}, no_cursor_timeout=True, session=session)
                pbar = tqdm(total=total_docs, desc=f"Indexing: {col}", unit="doc")
                batch = []

                for doc in cursor:
                    try:
                        action = normalize(doc, col)
                        batch.append(action)
                    except Exception as e:
                        logging.error(f"Error processing document in collection {col}: {doc}")
                        report_dict["collection"][col]["errors"] += 1
                        pbar.update(1)
                        pbar.set_postfix(errors=report_dict["collection"][col]["errors"])
                        continue

                    if len(batch) >= BATCH_SIZE:
                        try:
                            helpers.bulk(es, batch)
                            report_dict["collection"][col]["indexed_docs"] += len(batch)
                        except Exception as e:
                            logging.error(f"Error indexing collection {col}: {e}")
                            report_dict["collection"][col]["errors"] += len(batch)
                            with open("error_docs.json", "a", encoding="utf-8") as error_file:
                                json.dump(batch, error_file, indent=2, ensure_ascii=False)
                                error_file.write("\n")
                        finally:
                            pbar.update(len(batch))
                            pbar.set_postfix(errors=report_dict["collection"][col]["errors"])
                            batch = []

                # Final batch
                if batch:
                    try:
                        helpers.bulk(es, batch, raise_on_error=True)
                        report_dict["collection"][col]["indexed_docs"] += len(batch)
                    except Exception as e:
                        logging.error(f"Error indexing collection {col}: {e}")
                        report_dict["collection"][col]["errors"] += len(batch)
                        with open("error_docs.json", "a", encoding="utf-8") as error_file:
                            json.dump(batch, error_file, indent=2, ensure_ascii=False)
                            error_file.write("\n")
                    finally:
                        pbar.update(len(batch))
                        pbar.set_postfix(errors=report_dict["collection"][col]["errors"])

                pbar.close()

            except Exception as e:
                logging.error(f"Error indexing collection {col}: {e}")
                report_dict["collection"][col]["errors"] += 1
                continue

    with open("index_report.json", "a", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    tqdm.write("🎉 All collections indexed.")
    tqdm.write("📄 Report saved to 'index_report.json'.")

if __name__ == "__main__":
    index_to_elasticsearch()