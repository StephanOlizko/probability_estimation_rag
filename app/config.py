class Config():
    def __init__(self):

        #self.data_path = "data/clean_data.csv"
        #self.encoder_types = ["sentence_transformer", "tfidf"]
        #self.index_type = "faiss"
        #self.query = "Will a recession begin in the United States in the following years?"
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.k = 10
        self.fetching_links_num = 100
        self.config_path = "metrics/.log"