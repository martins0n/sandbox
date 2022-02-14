import logging

logging.basicConfig(format="%(process)d - %(asctime)s - %(filename)s - %(message)s")

_logger = logging.getLogger()

class _Logger():
    def log(self, message: str):
        _logger.error(f"{message} {id(self)}", stacklevel=2)


logger = _Logger()


from multiprocessing import Manager

d = Manager().dict()