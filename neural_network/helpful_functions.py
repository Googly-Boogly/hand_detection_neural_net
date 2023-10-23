# Sensitive info ln 100

import logging
# import pymysql.cursors
import os
import traceback
# from loguru import logger
# This will be a universal page, for use across multiple projects


def log_it(logger, error):
    """
    Takes in logger and error and logs the error in the correct way
    :param logger: logger object made in one of the logger maker functions
    :param error: the error that occured
    :return: None
    """
    log_message = f"An exception occurred on line {traceback.extract_tb(error.__traceback__)[-1].lineno}: {error}"
    logger.error(log_message)


def create_logger_error(file_path, name):
    """

    :param file_path: do this os.path.abspath(__file__)
    :param name: (str)
    :return:
    To use the logger
    from global_code.helpful_functions import create_logger_error, log_it
    logger = create_logger_error(os.path.abspath(__file__), '')
    try:
        x = 0/0
    except Exception as e:
        log_it(logger, e)
    """

    caller_dir = os.path.dirname(os.path.abspath(file_path))


    logs_dir = os.path.join(caller_dir, 'logs')

    os.makedirs(logs_dir, exist_ok=True)

    # Create a subfolder named after the calling file (without extension)
    calling_file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_logs_dir = os.path.join(logs_dir, calling_file_name)

    os.makedirs(file_logs_dir, exist_ok=True)


    log_file = os.path.join(file_logs_dir, f'{name}.log')


    logger = logging.getLogger(name)


    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger


# logging.basicConfig(level=logging.INFO, filename='log.log', filemode='w',
#                         format="%(asctime)s - %(levelname)s - %(message)s")



def create_logger_all(file_name, name):
    # ---------------------------------------------------------------------------
    # This will create a logger object that will display everything
    # ---------------------------------------------------------------------------

    logger = logging.getLogger(name)

    caller_dir = os.path.dirname(os.path.abspath(file_name))
    logs_dir = os.path.join(caller_dir, "../../utils/logs")
    os.makedirs(logs_dir, exist_ok=True)  # Create the "logs" folder if it doesn't exist

    log_file = os.path.join(logs_dir, f"{name}.log")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


class CustomError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# this class will give us an instance of a connection to our database
# class MySQLConnection:
#     def __init__(self, db):
#         # change the user and password as needed
#         connection = pymysql.connect(host='localhost',
#                                      port=8889,
#                                      user='root',
#                                      password='root',
#                                      db=db,
#                                      charset='utf8mb4',
#                                      cursorclass=pymysql.cursors.DictCursor,
#                                      autocommit=True)
#         # establish the connection to the database
#         self.connection = connection
#
#     # the method to query the database
#     def query_db(self, query, data=None):
#         with self.connection.cursor() as cursor:
#             try:
#                 query = cursor.mogrify(query, data)
#                 # On off SQL Text buttton
#                 print(query)
#                 cursor.execute(query, data)
#                 if query.lower().find("insert") >= 0:
#                     # INSERT queries will return the ID NUMBER of the row inserted
#                     self.connection.commit()
#                     return cursor.lastrowid
#                 elif query.lower().find("select") >= 0:
#                     # SELECT queries will return the data from the database as a LIST OF DICTIONARIES
#                     result = cursor.fetchall()
#                     return result
#                 else:
#                     # UPDATE and DELETE queries will return nothing
#                     self.connection.commit()
#             except Exception as e:
#                 # if the query fails the method will return FALSE
#                 print(e)
#                 return False
#             finally:
#                 # close the connection
#                 self.connection.close()
#             # connectToMySQL receives the database we're using and uses it to create an instance of MySQLConnection
#
#
# def connecttomysql(db):
#     return MySQLConnection(db)


def count_lines_of_code(directory):
    """
    Needs to be the absolute path of the directory also will exclude the venv folder
    :param directory: (absolute path of the directory)
    :return: total_lines of python in project
    """

    total_lines = 0

    for root, dirs, files in os.walk(directory):
        # Exclude the "venv" folder
        if "venv" in dirs:
            dirs.remove("venv")

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    total_lines += len(lines)

    return total_lines