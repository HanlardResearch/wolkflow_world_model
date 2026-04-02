import logging
import yaml
import os
import datetime


class LogManager:
    _instance = None
    _session_folder_path = None
    _session_task_name = None

    def __new__(cls, config_path=None, task_name=None):
        if cls._instance is not None:
            cls._instance._cleanup()
        cls._instance = super(LogManager, cls).__new__(cls)
        cls._instance._initialize(config_path, task_name)
        return cls._instance

    def _initialize(self, config_path, task_name):
        self.loggers = {}
        self.global_config = yaml.safe_load(open(config_path, "r"))
        self.task_name = task_name
        self.task_folder_path = self._create_task_folder()
        self.session_folder_path = self._get_or_create_session_folder()
        self.folder_path = self._create_log_folder()
        self._setup_main_logger()
        self._setup_model_logger()
        self._setup_training_logger()

    def _create_task_folder(self):
        task_folder_path = os.path.abspath(os.path.join(self.global_config.get('logging').get('logpath'), self.task_name))
        os.makedirs(task_folder_path, exist_ok=True)
        return task_folder_path

    def _get_or_create_session_folder(self):
        if LogManager._session_folder_path is None or LogManager._session_task_name != self.task_name:
            session_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            session_folder_path = os.path.join(self.task_folder_path, session_name)
            os.makedirs(session_folder_path, exist_ok=True)
            LogManager._session_folder_path = session_folder_path
            LogManager._session_task_name = self.task_name
        else:
            session_folder_path = LogManager._session_folder_path

        os.makedirs(session_folder_path, exist_ok=True)
        return session_folder_path

    def _create_log_folder(self):
        run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        existing_runs = [
            name for name in os.listdir(self.session_folder_path)
            if os.path.isdir(os.path.join(self.session_folder_path, name))
        ]

        max_index = 0
        for run_name in existing_runs:
            prefix = run_name.split("-", 1)[0]
            if prefix.isdigit():
                max_index = max(max_index, int(prefix))

        run_index = max_index + 1
        folder_name = f"{run_index}-{run_timestamp}"
        folder_path = os.path.join(self.session_folder_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def _setup_main_logger(self):
        main_logger = logging.getLogger('global')
        main_logger.setLevel(self.global_config.get('logging').get('level'))
        os.makedirs(self.folder_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.folder_path, "meta.log"), encoding="utf-8")
        fh.setLevel(self.global_config.get('logging').get('level'))

        formatter = logging.Formatter('[%(asctime)s %(levelname)s]\n%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
        fh.setFormatter(formatter)

        main_logger.addHandler(fh)
    
    def _setup_model_logger(self):
        model_logger = logging.getLogger('model')
        model_logger.setLevel(self.global_config.get('logging').get('level'))
        os.makedirs(self.folder_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.folder_path, "model_query.log"), encoding="utf-8")
        fh.setLevel(self.global_config.get('logging').get('level'))

        formatter = logging.Formatter('[%(asctime)s %(levelname)s]\n%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
        fh.setFormatter(formatter)

        model_logger.addHandler(fh)

    def _setup_training_logger(self):
        training_logger = logging.getLogger('train')
        training_logger.setLevel(self.global_config.get('logging').get('level'))
        os.makedirs(self.folder_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.folder_path,"train.log"), encoding="utf-8")
        fh.setLevel(self.global_config.get('logging').get('level')) 

        formatter = logging.Formatter('[%(asctime)s %(levelname)s]\n%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
        fh.setFormatter(formatter)

        training_logger.addHandler(fh)

    def create_logger(self, name, log_file, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        log_file = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        if not logger.handlers:
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s]\n%(message)s', datefmt='%Y-%d-%m %H:%M:%S'))
            logger.addHandler(handler)
        logger.propagate = False
        self.loggers[name] = logger

    def get_logger(self, index):
        return self.loggers.get(index, logging.getLogger())
    def _cleanup(self):
        for logger in self.loggers.values():
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        
        main_logger = logging.getLogger('global')
        handlers = main_logger.handlers[:]
        for handler in handlers:
            try:
                handler.close()
                main_logger.removeHandler(handler)
            except Exception as e:
                print(f"Error closing handler: {e}")
        
        model_logger = logging.getLogger('model')
        handlers = model_logger.handlers[:]
        for handler in handlers:
            try:
                handler.close()
                model_logger.removeHandler(handler)
            except Exception as e:
                print(f"Error closing handler: {e}")

        training_logger = logging.getLogger('train')
        handlers = training_logger.handlers[:]
        for handler in handlers:
            try:
                handler.close()
                training_logger.removeHandler(handler)
            except Exception as e:
                print(f"Error closing handler: {e}")
