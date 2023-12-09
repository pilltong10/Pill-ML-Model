import json

# JSON 파일을 불러오기
def pill_getname(drug_N):
    with open('integrated_data_v2.json', 'r', encoding='utf-8') as file:
        data = json.load(file)


    # 데이터를 반복하면서 원하는 drug_N을 찾고 해당 객체의 dl_name 값을 얻음
    found_dl_name = None
    for item in data['images']:
        if item['drug_N'] == drug_N:
            found_dl_name = item['dl_name']
            break

    if found_dl_name:
        return found_dl_name
    else:
        return "Not found"