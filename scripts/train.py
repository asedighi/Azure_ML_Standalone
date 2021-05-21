import argparse
import logging
import debugpy
from azureml.core import Run
#from azdebugrelay import DebugRelay, DebugMode, debugpy_connect_with_timeout


def _main():
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store',
                        default="", choices=['attach', 'none'], required=False)
    parser.add_argument('--debug-relay-connection-string-secret', action='store',
                        default="", required=False)
    parser.add_argument('--debug-relay-connection-name', action='store',
                        default="", required=False)
    parser.add_argument('--debug-port', action='store', type=int,
                        default=5678, required=False)
    options, _ = parser.parse_known_args()
    

    run = Run.get_context()
    debug_relay = None
    debug = False

    if options.debug == "attach":
        if options.debug_relay_connection_string_secret == "" or options.debug_relay_connection_name == "":
            err_msg = "Azure Relay connection string secret name or connection name is empty."
            logging.fatal(err_msg)
            raise ValueError(err_msg)
        # get connection string from the workspace Key Vault
        connection_string = run.get_secret(
            options.debug_relay_connection_string_secret)
        if connection_string is None or connection_string == "":
            err_msg = "Connection string for Azure Relay Hybrid Connection is missing in Key Vault."
            logging.fatal(err_msg)
            raise ValueError(err_msg)
        debug = True
        relay_connection_name = options.debug_relay_connection_name # your Hybrid Connection name
        debug_mode = DebugMode.Connect
        hybrid_connection_url = None # can keep it None because using a connection string
        host = "127.0.0.1"  # local hostname or ip address the debugger starts on
        port = options.debug_port
        debugpy_timeout = 15

        debug_relay = DebugRelay(
            connection_string, relay_connection_name, debug_mode, hybrid_connection_url, host, port)
        debug_relay.open()
        print(f"Starting debugpy session on {host}:{port}")
        if debugpy_connect_with_timeout(host, port, debugpy_timeout):
            print("Debugpy is connected!")
        else:
            print("Debugpy could not connect!")
    '''
    train_job()

    '''
    if debug_relay is not None:
        debug_relay.close()
    '''

def train_job(debug: bool = False):
    """This is supposed to be a function with traning code.
    We have a breakpoint here!

    Args:
        debug (bool, optional): Debugging mode. Defaults to False.
    """
   
    print(f"Doing my work. Debug mode is {debug}.")


##########################
if __name__ == '__main__':
    _main()