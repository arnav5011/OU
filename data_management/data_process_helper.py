from datetime import datetime
def parse_date(date):
    days = date%100   
    date = date // 100
    month = date%100
    year = date // 100
    return datetime(year, month, days, 0, 0, 0)