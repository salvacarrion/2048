import urllib, json, threading, itertools
import urllib.request

try:
    import websocket
except ImportError:
    websocket = None


class ChromeDebuggerControl(object):
    """ Control Chrome using the debugging socket.
    Chrome must be launched using the --remote-debugging-port=<port> --remote-allow-origins="*" options for this to work! """

    def __init__(self, port):
        if websocket is None:
            raise NotImplementedError("websocket-client library not available; cannot control Chrome.\n"
                                      "Please install it (pip install websocket-client) then try again.")

        # Obtain the list of pages
        pages = json.loads(urllib.request.urlopen('http://localhost:%d/json/list' % port).read())
        tab_pages = [p for p in pages if p.get('type') == 'page']

        if len(tab_pages) == 0:
            raise Exception("No active browser tabs found! Make sure Chrome is open with at least one tab.")

        target_page = None
        for p in tab_pages:
            title = p.get('title', '')
            url = p.get('url', '')
            if '2048' in title or '2048' in url:
                target_page = p
                print(f"Attached to 2048 page: '{title}' ({url})")
                break

        if target_page is None:
            open_tabs_str = "\n".join([f" - '{p.get('title')}' ({p.get('url')})" for p in tab_pages])
            raise Exception(
                "Could not find any tab with '2048' in the title or URL.\n"
                f"Currently open tabs:\n{open_tabs_str}\n"
                "Please navigate to https://classic.play2048.co in your debug Chrome window."
            )

        page = target_page

        # Configure debugging websocket
        wsurl = page['webSocketDebuggerUrl']
        self.ws = websocket.create_connection(wsurl)

        self.requests = {} # dictionary containing in-flight requests
        self.results = {}
        self.req_counter = itertools.count(1)

        self.thread = threading.Thread(target=self._receive_thread)
        self.thread.daemon = True
        self.thread.start()

        self._send_cmd_noresult('Runtime.enable')

    def _receive_thread(self):
        """ Continually read events and command results """
        while 1:
            try:
                message = json.loads(self.ws.recv())
                if 'id' in message:
                    id = message['id']
                    event = self.requests.pop(id, None)
                    if event is not None:
                        self.results[id] = message
                        event.set()
            except Exception as e:
                break

    def _send_cmd_noresult(self, method, **params):
        """ Send a command and ignore the result. """
        id = next(self.req_counter)
        out = {'id': id, 'method': method}
        if params:
            out['params'] = params
        self.ws.send(json.dumps(out))

    def _send_cmd(self, method, **params):
        """ Send a command and wait for the result to be available. """
        id = next(self.req_counter)
        out = {'id': id, 'method': method}
        if params:
            out['params'] = params

        # Receive thread will signal us when the response is available
        event = threading.Event()
        self.requests[id] = event
        self.ws.send(json.dumps(out))
        event.wait()

        resp = self.results.pop(id)
        if 'error' in resp:
            raise Exception("Command %s(%s) failed: %s (%d)" % (
                method, ', '.join('%s=%r' % (k,v) for k,v in params.iteritems()), resp['error']['message'], resp['error']['code']))
        return resp['result']

    def execute(self, cmd):
        resp = self._send_cmd('Runtime.evaluate', expression=cmd)
        if 'exceptionDetails' in resp:
            raise Exception("JS evaluation threw an error: %s" % resp['result']['description'])
        result = resp['result']
        if 'value' in result:
            return result['value']
        if 'description' in result:
            return result['description']
        return None
