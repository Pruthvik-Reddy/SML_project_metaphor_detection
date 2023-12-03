from data_loader import get_data
import pandas as pd


if __name__=="__main__":
    data=get_data()
    print(data.shape)