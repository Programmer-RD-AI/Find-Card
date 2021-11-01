# from sklearn.model_selection import ParameterGrid
# param_grid = {'param1': ['value1', 'value2', 'value3'], 'paramN' : ['value1', 'value2', 'valueM']}

# grid = ParameterGrid(param_grid)

# for params in grid:
#     print(params)

# import pandas as pd
# print(type(pd.read_csv('./V2/Data.csv')))
# print(type())


# from barcode import EAN13

# # Make sure to pass the number as string
# number = '094718024596'

# # Now, let's create an object of EAN13
# # class and pass the number
# my_code = EAN13(number)

# # Our barcode is ready. Let's save it.
# my_code.save("iartmart-1")

# import qrcode
# from PIL import Image
# img = qrcode.make('https://iartmart.com/')
# qr = qrcode.QRCode(
#     version=1,
#     error_correction=qrcode.constants.ERROR_CORRECT_H,
#     box_size=10,
#     border=4,
# )
# qr.add_data('https://iartmart.com/')
# qr.make(fit=True)
# img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
# img.save("iartmart-2.png")

def to_jaden_case(string):
    string = string.split(' ')
    new_string_list = []
    for s in string:
        # s[0] = s[0].capitalize()
        print(s.split(''))


to_jaden_case("How can mirrors be real if our eyes aren't real")
