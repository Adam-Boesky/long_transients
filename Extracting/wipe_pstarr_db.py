"""Script for wiping the panstarrs database. I use this when I edit the Pan-STARRS SQL extraction query. Make sure the 
user in Extracting/utils.py is the right one before running."""
from utils import MASTCASJOBS


def wipe_db():
    for table in MASTCASJOBS.list_tables():
        print(f"Dropping table {table}")
        MASTCASJOBS.drop_table(table)

    assert len(MASTCASJOBS.list_tables()) == 0, "Failed to wipe database!!!"


if __name__ == '__main__':
    wipe_db()
