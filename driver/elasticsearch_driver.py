import logging

from elasticsearch import Elasticsearch, helpers

LOGGER = logging.getLogger("driver")


class ElasticSearchDriver:
    def __init__(self, hosts):
        self._es = Elasticsearch(hosts,
                                 http_compress=True,
                                 send_get_body_as="POST")

    def query(self, index, body):
        res = self._es.search(body=body, index=index)
        return res['hits']['hits']

    def _insert(self, index, data):
        try:
            return helpers.bulk(self._es, data, index=index)
        except Exception as e:
            
            LOGGER.exception(e)
            return None

    def update_record(self, index, id_record, data):
        try:
            return self._es.update(index=index, id=id_record, body=data)
        
        except Exception as e:
            LOGGER.exception(e)
            return None

    def _check_exists(self, index, data):
    
        try:
            res = self._es.search(index=index, body=data)
            return res['hits']['total']['value'] > 0
        
        except Exception as e:
            LOGGER.exception(e)
            return None
        
    def _check_exists_embedding(self,index,data):
        record = []
        try:
            res = self._es.search(index, body=data)
            if res['hits']['total']['value'] > 0:
                record = res['hits']['hits'][0]['_id']
            return res['hits']['total']['value'] > 0, record
        except Exception as e:
            LOGGER.exception(e)
            return None
