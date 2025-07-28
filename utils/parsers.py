from datetime import datetime
def parse_date(date):
    days = date%100
    if days < 10: days = f"0{days}"
    date = date // 100
    month = date%100
    if month < 10: month = f"0{month}"
    year = date // 100
    return f"{year}-{month}-{days}"


print(parse_date(20100101))