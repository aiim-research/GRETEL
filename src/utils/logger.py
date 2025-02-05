import sys
import os
import socket

# import picologging as logging
import logging


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

        os.makedirs(GLogger._path, exist_ok=True)

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


class PicklableGLogger:
    _instance = None  # Store instance separately
    _path = "output/logs"
    _logger_name = "PicklableGLogger"  # Unique logger name

    def __init__(self):
        """ Private constructor. Use getLogger() instead of creating an instance directly. """
        self._init_logger()

    def _init_logger(self):
        """ Initialize the logger with file and stream handlers. """
        self.info = logging.getLogger(self._logger_name)
        if not self.info.hasHandlers():  # Prevent duplicate handlers
            self.info.setLevel(logging.INFO)

            os.makedirs(PicklableGLogger._path, exist_ok=True)

            log_filename = f"{PicklableGLogger._path}/{os.getenv('JOB_ID', str(os.getpid()))}-{socket.gethostname()}.info"
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            stdout_handler = logging.StreamHandler(sys.stdout)

            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(process)d - %(message)s")
            file_handler.setFormatter(fmt)
            stdout_handler.setFormatter(fmt)

            self.info.addHandler(stdout_handler)
            self.info.addHandler(file_handler)

    @classmethod
    def getLogger(cls):
        """ Ensures a singleton logger instance. """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.info  # Return the actual logger object

    def __getstate__(self):
        """ Remove non-pickleable attributes before serialization. """
        return {"_logger_name": self._logger_name}

    def __setstate__(self, state):
        """ Reinitialize the logger when unpickling. """
        self._logger_name = state["_logger_name"]
        self._init_logger()  # Reinitialize the logger