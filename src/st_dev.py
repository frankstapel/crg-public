from math import floor
from os import path, stat
import gspread
import pickle
from oauth2client.service_account import ServiceAccountCredentials


def main():
    for index, st_dev in enumerate(get_st_dev()):
        print("Difficulty bucket [" + str(index) + ", " + str(index + 1) + ") has an average st. dev. of " + str(st_dev)[:5])


def get_st_dev():
    # to force refresh the st_dev data, delete the file ../misc/st_dev.pkl
    if not path.exists('../misc/st_dev.pkl'):
        open('../misc/st_dev.pkl', 'a').close()
    if stat('../misc/st_dev.pkl').st_size != 0:
        with open('../misc/st_dev.pkl', 'rb') as st_dev_input:
            st_dev = pickle.load(st_dev_input)
    else:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            '../misc/Climbing Route Generation-f8a5fdbc71a5.json', scope)
        gc = gspread.authorize(credentials)
        difficulties = gc.open_by_key('1pwnKZJPTzYeM6ImzxG5_9y8vLy969lIaJesnB_LVEQ0').worksheet('python_difficulties').get_all_records()
        deviations = gc.open_by_key('1pwnKZJPTzYeM6ImzxG5_9y8vLy969lIaJesnB_LVEQ0').worksheet('python_stdevs').get_all_records()

        # Setup a st_dev of 11 wide and 18 high
        st_dev = [0.0 for _ in range(11)]
        amount = [0.0 for _ in range(11)]

        print(difficulties[0])
        print(deviations[0].get(str(0)))

        # Go over each row of the MoonBoard
        for column in range(11):  # type: int
            # Go over each column of the MoonBoard
            for row in range(18):  # type: int
                index = floor(float(str(difficulties[row].get(str(column))).replace(',', '.')))
                st_dev[index] += float(str(deviations[row].get(str(column))).replace(',', '.'))
                amount[index] += 1.0

        for i in range(11):
            st_dev[i] = st_dev[i] / amount[i]

        with open('../misc/st_dev.pkl', 'wb') as st_dev_output:
            pickle.dump(st_dev, st_dev_output, pickle.HIGHEST_PROTOCOL)
    return st_dev


if __name__ == '__main__':
    main()
