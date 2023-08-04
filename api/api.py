import sys
import os
import cv2
import io
import base64
import requests
import numpy as np
from numpy.linalg import norm
from PIL import Image
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
from elasticsearch import Elasticsearch, helpers


sys.path.append("/home/tima/detec_and_tracking/Face-Mask-Detection")
from run_face_mask import face_mask_end2end
sys.path.append("/home/tima/detec_and_tracking/driver")
from driver.elasticsearch_driver import *

prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").rstrip("/")

face_mask = face_mask_end2end()

es = ElasticSearchDriver("http://0.0.0.0:9200/")

app = FastAPI(
    title = "Face_Mask",
    version = "1.0.0",
    descripton = "Detect and draw mask",
    openapi_prefix = prefix,
    )
 
@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"{prefix}/docs")

dfloat32 = np.dtype('>f4')
def encode_array(arr):
    base64_str = base64.b64encode(np.array(arr).astype(dfloat32)).decode("utf-8")
    return base64_str

async def file_to_image(file):
    npimg = np.frombuffer(await file.read(), np.int8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

async def url_to_image(img_str):
    imgdata = requests.get(img_str)    
    img = Image.open(io.BytesIO(imgdata.content))
    img = np.asarray(img)
    return img

def predict(img):
    f1,f2 = face_mask.extract_feature(img)
    return f1, f2

@app.post("/api/get_image")
async def face_mask_route(name: str, phone: str, files: UploadFile = File(...)):
    try:
        image = await file_to_image(files)
        feature_face, feature_mask = predict(image)
        vec = (feature_face[0] / norm(feature_face[0])).tolist()
        query = {
            "min_score": 0.95,
            "query": {
                "function_score": {
                    "boost_mode": "replace",
                    "script_score": {
                        "script": {
                            "source": "binary_vector_score",
                            "lang": "knn",
                            "params": {
                                "cosine": True,
                                "field": "face_embedding",
                                "vector": vec
                            }
                        }
                    }
                }
            },
            "_source": ['name','phone']
        }
        
        check_emb, id_exist = es._check_exists_embedding(index='id_face', data = query)
        
        if check_emb:
            try:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"name": name}},
                                {"match": {"phone": phone}}
                            ]
                        }
                    }
                }
                
                check_exists = es._check_exists(index='id_face', data = query)
                
                if check_exists:
                    return JSONResponse(status_code=200, content={"message": "Already Exists"})
                else:
                    update_data = {
                        "doc": {
                            "name": name,
                            "phone": phone
                        }
                    }
                    es.update_record(index="id_face", id_record=id_exist, data=update_data)
                    
                    return JSONResponse(status_code=200, content={"message": "Success"})
                
            except Exception as e:
                return JSONResponse(content={"message": f"Error while checking existence: {str(e)}"}, status_code=500)
        else:
            data = [
                {
                    'name': name,
                    'phone': phone,
                    'create_on': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                    'face_embedding': encode_array(feature_face[0]),
                    'face_mask_embedding': encode_array(feature_mask[0]),
                    'last_modified': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                },
            ]
            
            try:
                es._insert(index = 'id_face', data = data)
                return JSONResponse(status_code=200, content={"message": "Success"})
            
            except Exception as e:
                return JSONResponse(content={"message": f"Error while bulk indexing: {str(e)}"}, status_code=500)
            
    except Exception as e:
        return JSONResponse(content={"message": f"Error while processing image: {str(e)}"}, status_code=500)


if __name__=='__main__':
    uvicorn.run('api:app',reload=True)