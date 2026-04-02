from tools.base.register import global_tool_registry
from tools.base.base_tool import Tool
from abc import abstractmethod
from tools.utils.broswer import SimpleTextBrowser
import signal
from functools import wraps

def timeout_handler(signum, frame):
    raise TimeoutError("Request timed out")

def timeout(seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the signal handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

class Web_Search(Tool):
    def __init__(self):
        super().__init__("web_search", "Search the web for a given query", self.execute)
        self.broswer = SimpleTextBrowser()
    
    def execute(self,*args,**kwargs):
        try: 
            query = kwargs.get("query", "")
            self.broswer.downloads_folder = kwargs.get("work_path", "")
            flag, ans = self.search(query)
        except AttributeError:
            return False, "No results found for query {}".format(query)
        except TimeoutError:
            return False, "Timeout"
        except Exception as e:
            return False, "No results found for query {}".format(query)

        if (ans is None) or (len(ans) == 0):
            # raise ValueError(f"No results found for query {query}.")
            return False, "No results found for query {}".format(query)
        
        return flag, ans
    
    @abstractmethod
    def search(self, query):
        pass

@global_tool_registry("access_website")
class Website_SearchEngine(Web_Search):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def search(self, url):
        self.broswer.set_address(url)
        if self.broswer.page_content != None and len(self.broswer.page_content) != 0:
            if "Failed to fetch" in self.broswer.page_content:
                return False, self.broswer.page_content
            else:
                return True, self.broswer.page_content
        else:
            return False, "Can not Access this website: {}".format(url)
if __name__ == '__main__':
    url = "https://tutorial.math.lamar.edu/Classes/DE/Modeling.aspx"
    agent = Website_SearchEngine("web_access")
    x,y= agent.search(url="https://www.baidu.com")
    print(x)
    print(y)