import sys
import picologging as logging
import os
import socket


class GLogger(object):
    __create_key = object()
    __logger = None
    _path = "output/logs"

    def __init__(self, create_key):
        assert(create_key == GLogger.__create_key), \
            "GLogger objects must be created using GLogger.getLogger"
        self.real_init()
        
    def real_init(self):
        self.info = logging.getLogger()
        self.info.setLevel(logging.INFO)

        if not os.path.exists(GLogger._path):
            os.makedirs(GLogger._path)

        file_handler = logging.FileHandler(GLogger._path+"/"+str(os.getenv('JOB_ID',str(os.getpid())))+"-"+socket.gethostname()+".info", encoding='utf-8')
        stdout_handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(process)d - %(message)s"
        )

        file_handler.setFormatter(fmt)
        stdout_handler.setFormatter(fmt)
        
        self.info.addHandler(stdout_handler)
        self.info.addHandler(file_handler)
        
    @classmethod
    def getLogger(self):
        if(GLogger.__logger == None):
            GLogger.__logger = GLogger(GLogger.__create_key)
            
        return GLogger.__logger.info

        
#### EXAMPLE OF USAGE ####
#from src.utils.logger import GLogger
#GLogger._path="log" #Set the directory only once

#logger = GLogger.getLogger()
#logger.info("Successfully connected to the database '%s' on host '%s'", "my_db", "ubuntu20.04")        
#logger.warning("Detected suspicious activity from IP address: %s", "111.222.333.444")