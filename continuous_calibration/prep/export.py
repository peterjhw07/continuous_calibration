import pandas as pd


def export_xlsx(df, filename, mode="w", if_sheet_exists=None, sheet_name='Sheet1'):
    if '.xlsx' not in filename:
        filename += '.xlsx'
    with pd.ExcelWriter(filename, mode=mode, if_sheet_exists=if_sheet_exists, engine='openpyxl') as writer:

        data_store_try = False
        while data_store_try is False:
            try:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                data_store_try = True
            except PermissionError:
                input('Error! Export file open. Close and then press enter.')
            except:
                print('Unknown error! Check inputs are formatted correctly. Else examine error messages and review code.')
