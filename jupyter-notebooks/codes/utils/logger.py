import logging

class logger():
    """
    filename이 None이면 파일을 기록하지 않음. 
    append mode는 기존 파일을 덮어씌우지 않고 로그를 이어붙임. 
    instance를 2개 이상 생성해서 동일한 파일에 작성한다면 서로 혼란을 줄 수 있음.
    """
    def __init__(self, filename=None, append_mode=False):
        self.logger = logging.getLogger() # root logger
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] >> %(message)s')

        # stream handler
        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(logging.DEBUG)
        streamhandler.setFormatter(formatter)
        self.logger.addHandler(streamhandler)
        
        # file handler
        if filename != None:
            if append_mode is False:
                filehandler = logging.FileHandler(filename + '.txt', mode='w')
            else:
                filehandler = logging.FileHandler(filename + '.txt')
            filehandler.setLevel(logging.DEBUG)
            filehandler.setFormatter(formatter)
            self.logger.addHandler(filehandler)

        # initial logs
        if filename == None:
            self.logger.warning('Logs will be recorded at stream only.')
        else:
            self.logger.warning('Logs will be recorded at stream and txt file using \'append_mode={}\''.format(append_mode))
            

    def debug(self, data):
        self.logger.debug(data)


    def info(self, data):
        self.logger.info(data)
    

    def warning(self, data):
        self.logger.warning(data)

