# -*- coding: utf-8 -*-
import os

BASE_DIR = os.environ.get("STAT8307_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.environ.get("STAT8307_DATA", os.path.join(os.path.dirname(BASE_DIR), "data.csv"))
OUTPUT_DIR = BASE_DIR
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
NEWSAPI_KEY = "9661e7bb9bd24ea7bcc31e862e1636be"
