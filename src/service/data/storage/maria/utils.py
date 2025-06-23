from typing import List

import mariadb


def require_existing_dataset(func):
    """Annotation to assert that a given function requires a valid dataset name as the first non-self argument"""
    async def validate_dataset_exists(*args, **kwargs):
        storage, dataset_name = args[0], args[1]
        if not await storage.dataset_exists(dataset_name):
            raise ValueError(f"Error when calling {func.__name__}: Dataset '{dataset_name}' does not exist.")
        return await func(*args, **kwargs)

    return validate_dataset_exists



def get_clean_column_names(column_names) -> List[str]:
    """
    Programmatically generate the column names in a model data table.
    This avoids possible SQL injection from the real column names coming from the mode.
    """
    return [f"column_{i}" for i in range(len(column_names))]


class MariaConnectionManager:
    def __init__(self, user, password, host, port, database):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

    def __enter__(self):
        try:
            self.conn = mariadb.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database
            )
        except mariadb.Error as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            raise e
        return self.conn, self.conn.cursor()

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()