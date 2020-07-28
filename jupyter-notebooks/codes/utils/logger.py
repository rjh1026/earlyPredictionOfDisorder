import logging

class logger():
    """
    append mode는 기존 파일을 덮어씌우지 않고 로그를 이어붙임.
    instance를 2개 이상 생성해서 동일한 파일에 작성한다면 서로 혼란을 줄 수 있음.
    """
    def __init__(self, filename, append_mode):
        self.logger = logging.getLogger() # root logger
        self.logger.setLevel(logging.DEBUG)

        # stream handler
        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(logging.DEBUG)
        
        # file handler
        if append_mode is False:
            filehandler = logging.FileHandler(filename + '.txt', mode='w')
        else:
            filehandler = logging.FileHandler(filename + '.txt')
        filehandler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] >> %(message)s')
        streamhandler.setFormatter(formatter)
        filehandler.setFormatter(formatter)

        self.logger.addHandler(streamhandler)
        self.logger.addHandler(filehandler)

        # initial logs
        self.logger.warning('\n')
        self.logger.warning('Text logger created with \'append_mode={}\''.format(append_mode))
        

    def print_debug(self, data):
        self.logger.debug(data)


    def print_info(self, data):
        self.logger.info(data)
    

    def print_warning(self, data):
        self.logger.warning(data)

