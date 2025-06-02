import pandas as pd


# Read imported data
def raw_import(filename, sheet_name='Sheet1', t_col=0, col=1):
    """
    Read in data from excel filename

    Params
    ------

    Returns
    -------


    """
    if '.xlsx' in filename:
        df = pd.read_excel(filename, sheet_name=sheet_name, engine='openpyxl', dtype=str)
        headers = list(pd.read_excel(filename, sheet_name=sheet_name, engine='openpyxl').columns)
    elif '.txt' in filename:
        df = pd.read_csv(filename)
        headers = list(pd.read_csv(filename).columns)
    else:
        try:
            df = pd.read_excel(filename, sheet_name=sheet_name, engine='openpyxl', dtype=str)
            headers = list(pd.read_excel(filename, sheet_name=sheet_name, engine='openpyxl').columns)
        except Exception as e:
            raise e
    if isinstance(col, int): col = [col]
    conv_col = [i for i in [t_col, *col] if i is not None]
    try:
        for i in conv_col:
            df[headers[i]] = pd.to_numeric(df[headers[i]], downcast='float')
        return df
    except ValueError:
        raise ValueError('Excel file must contain data rows (i.e. col specified) of numerical input with at most 1 header row.')
