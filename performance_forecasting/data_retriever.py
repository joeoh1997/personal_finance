import requests
import pandas as pd

def request(url):
    response = requests.get(url)
    return response.json()


def standard_industrial_classification(ticker, apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"):
    return request(
        "https://financialmodelingprep.com/api/v4/standard_industrial_classification"
        f"?symbol={ticker}&apikey={apikey}"
    )


def company_profile(ticker, apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"):
    return request(
        f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={apikey}"
    )


def cash_flow_growth(
    ticker, period="year", limit=1000, keep_cols=None, apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"
):

    if keep_cols is None:
        keep_cols = [
            "symbol", "date", "growthNetIncome", "growthDepreciationAndAmortization",
            "growthDeferredIncomeTax", "growthInvestmentsInPropertyPlantAndEquipment",
            "growthAcquisitionsNet", "growthPurchasesOfInvestments",
            "growthDebtRepayment", "growthNetChangeInCash",
            "growthCapitalExpenditure", "growthFreeCashFlow"
        ]

    data = get_data_df(
        "cash-flow-statement-growth",
        ticker, keep_cols, period,
        limit, apikey
    )

    return data


def cash_flow(
    ticker, period="year", limit=1000, keep_cols=None, apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"
):

    if keep_cols is None:
        keep_cols = [
            "date", "symbol", "netIncome", "depreciationAndAmortization",
            "deferredIncomeTax", "accountsReceivables", "accountsPayables", "inventory",
            "netCashProvidedByOperatingActivities", "netCashUsedProvidedByFinancingActivities",
            "netCashUsedForInvestingActivites", "investmentsInPropertyPlantAndEquipment",
            "acquisitionsNet", "purchasesOfInvestments", "salesMaturitiesOfInvestments",
            "debtRepayment", "commonStockIssued", "commonStockRepurchased",
            "dividendsPaid", "effectOfForexChangesOnCash",
            "cashAtEndOfPeriod", "capitalExpenditure", "freeCashFlow",
        ]

    data = get_data_df(
        "cash-flow-statement",
        ticker, keep_cols, period,
        limit, apikey
    )

    return data


def income_statement_growth(
    ticker, period="year", limit=1000, keep_cols=None, apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"
):

    if keep_cols is None:
        keep_cols = [
            "symbol", "date", "growthRevenue", "growthCostOfRevenue",
            "growthResearchAndDevelopmentExpenses","growthGeneralAndAdministrativeExpenses",
            "growthSellingAndMarketingExpenses", "growthOperatingExpenses",
            "growthInterestExpense", "growthDepreciationAndAmortization"
        ]

    data = get_data_df(
        "income-statement-growth",
        ticker, keep_cols, period,
        limit, apikey
    )

    return data


def income_statement(
    ticker, period="year", limit=1000, keep_cols=None, apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"
):

    if keep_cols is None:
        keep_cols = [
            "date", "symbol", "reportedCurrency",
            "revenue", "costOfRevenue", "researchAndDevelopmentExpenses",
            "generalAndAdministrativeExpenses", "sellingAndMarketingExpenses",
            "sellingGeneralAndAdministrativeExpenses", "operatingExpenses",
            "costAndExpenses", "interestExpense", "depreciationAndAmortization",
            "incomeTaxExpense", "netIncome",
        ]

    data = get_data_df(
        "income-statement",
        ticker, keep_cols, period,
        limit, apikey
    )

    return data


def balance_sheet_growth(
    ticker, period="year", limit=1000, keep_cols=None, apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"
):

    if keep_cols is None:
        keep_cols = [
            "symbol", "date", "growthShortTermInvestments", "growthLongTermInvestments",
            "growthNetReceivables", "growthAccountPayables",
            "growthPropertyPlantEquipmentNet", 
            "growthShortTermDebt", "growthLongTermDebt",
            "growthTotalLiabilities", "growthTotalAssets",
            "growthTotalInvestments", "growthTotalDebt",
        ]

    data = get_data_df(
        "balance-sheet-statement-growth",
        ticker, keep_cols, period,
        limit, apikey
    )

    return data


def balance_sheet(
    ticker, period="year", limit=1000, keep_cols=None, apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"
):

    if keep_cols is None:
        keep_cols = [
            "symbol", "date", "cashAndCashEquivalents",
            "shortTermInvestments", "cashAndShortTermInvestments",
            "netReceivables", "inventory", "propertyPlantEquipmentNet",
            "longTermInvestments", "shortTermDebt", "taxAssets",
            "accountPayables", "taxPayables", "deferredRevenue", "longTermDebt",
            "totalCurrentAssets", "totalNonCurrentAssets", "totalCurrentLiabilities",
            "totalNonCurrentLiabilities", "retainedEarnings", "accumulatedOtherComprehensiveIncomeLoss",
            "totalStockholdersEquity", "totalInvestments", "totalDebt", "netDebt",
        ]

    data = get_data_df(
        "balance-sheet-statement",
        ticker, keep_cols, period,
        limit, apikey
    )

    return data


def full_financial_statement(
    ticker, period="year", limit=1000, keep_cols=None, apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"
):

    if keep_cols is None:
        keep_cols = [
            "symbol", "date",
            "revenuefromcontractwithcustomerexcludingassessedtax", "comprehensiveincomenetoftax",
            "grossprofit", "costofgoodsandservicessold", "nonoperatingincomeexpense", "operatingexpenses",
            "researchanddevelopmentexpense", "sellinggeneralandadministrativeexpense", "incometaxespaidnet",
            "repaymentsoflongtermdebt", "paymentsforrepurchaseofcommonstock",
            "cashcashequivalentsrestrictedcashandrestrictedcashequivalents", "othernoncashincomeexpense",
            "paymentstoacquirebusinessesnetofcashacquired", "netcashprovidedbyusedinoperatingactivities",
            "proceedsfromsaleofavailableforsalesecuritiesdebt", "repaymentsoflongtermdebt", "incometaxespaidnet",
            "proceedsfromissuanceoflongtermdebt", "netcashprovidedbyusedininvestingactivities", "interestpaidnet",
            "netcashprovidedbyusedinfinancingactivities", "proceedsfromrepaymentsofcommercialpaper",
            "paymentstoacquireavailableforsalesecuritiesdebt", "paymentstoacquirepropertyplantandequipment",
            "paymentsofdividends", "increasedecreaseininventories", "increasedecreaseinaccountspayable",
            "increasedecreaseinaccountsreceivable", "increasedecreaseincontractwithcustomerliability",
            "proceedsfromissuanceofcommonstock", "depreciationdepletionandamortization",
            "proceedsfrommaturitiesprepaymentsandcallsofavailableforsalesecurities",
            "liabilitiescurrent", "liabilitiesnoncurrent", "propertyplantandequipmentnet",
            "commercialpaper", "longtermdebtcurrent", "longtermdebtnoncurrent", "stockholdersequity",
            "accountsreceivablenetcurrent", "accountspayablecurrent", "assetscurrent",
            "assetsnoncurrent", "inventorynet"
        ]

    data = get_data_df(
        "financial-statement-full-as-reported",
        ticker, keep_cols, period,
        limit, apikey
    )

    return data


def get_data_df(
        call_type,
        ticker,
        keep_cols,
        period="year",
        limit=1000,
        apikey="f09ef0f6985bef8f53ad5f0ed68dc30c"
):
    
    data = request(
        f"https://financialmodelingprep.com/api/v3/{call_type}/"
        f"{ticker}?limit={limit}&apikey={apikey}&period={period}"
    )
  
    data = pd.DataFrame(data)

    if keep_cols != "all":
        data = data[keep_cols]

    return data
    

if __name__ == "__main__":
    ticker='AAPL'
    join_cols = ["symbol", "date"]

    retrieval_functions = [
        balance_sheet, cash_flow, income_statement,
        balance_sheet_growth, cash_flow_growth,
        income_statement_growth
    ]

    data = retrieval_functions[0](ticker)

    for retrieval_function in retrieval_functions[1:]:
        data = pd.merge(
            data,
            retrieval_function(ticker), 
            on=join_cols
        )

    data['industryTitle'] = [standard_industrial_classification(ticker)[0]['industryTitle']] * len(data)

    profile = company_profile(ticker)[0]

    for var_name in [
          "sector", "country", "fullTimeEmployees", 
          "currency", "exchange", "industry", "isEtf", 
          "isActivelyTrading", "isAdr", "isFund"
    ]:
        data[var_name] = [profile[var_name]] * len(data)

    data.to_csv('test.csv')