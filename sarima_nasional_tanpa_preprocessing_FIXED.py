# ==========================================================
# SARIMA NASIONAL TANPA PREPROCESSING - VERSI FIXED
# ==========================================================
# Logika pemilihan model disamakan dengan ARIMA before:
# 1. Grid search
# 2. Uji Ljung-Box: p-value > 0.05
# 3. Hanya model yang lolos Ljung-Box masuk validation
# 4. Validation dilakukan pada 6 bulan terakhir dari data training
# 5. Model dipilih berdasarkan Validation MAE terkecil
# 6. Jika perlu pembanding, Validation RMSE lalu AIC
# 7. Model terpilih di-fit ulang pada seluruh 18 bulan training
# 8. Forecast 6 bulan testing
# 9. Evaluasi test + screening grafik
# ==========================================================

import os
import warnings
import itertools

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


# ==========================================================
# FUNGSI METRIK
# ==========================================================

def hitung_metrik(actual, prediksi):
    actual = pd.Series(actual, dtype=float)
    prediksi = pd.Series(prediksi, index=actual.index, dtype=float)

    valid = actual.notna() & prediksi.notna()
    actual = actual[valid]
    prediksi = prediksi[valid]

    if len(actual) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mae = mean_absolute_error(actual, prediksi)
    rmse = np.sqrt(mean_squared_error(actual, prediksi))

    mask_mape = actual != 0

    if mask_mape.sum() > 0:
        mape = (
            np.mean(
                np.abs(
                    (actual[mask_mape] - prediksi[mask_mape])
                    / actual[mask_mape]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    if np.abs(actual).sum() > 0:
        wape = (
            np.abs(actual - prediksi).sum()
            / np.abs(actual).sum()
        ) * 100
    else:
        wape = np.nan

    return mae, rmse, mape, wape


# ==========================================================
# FILE EXCEL
# ==========================================================

file_excel = "DATASET SARIMA SEBELUM PREPROCESSING.xlsm"
excel = pd.ExcelFile(file_excel)

# ==========================================================
# FOLDER HASIL
# ==========================================================

folder_output = "HASIL_GRAFIK_SARIMA"
os.makedirs(folder_output, exist_ok=True)

print("Jumlah sheet:", len(excel.sheet_names))
print()

for i, sheet in enumerate(excel.sheet_names, start=1):
    print(i, sheet)

print("\nSELESAI")


# ==========================================================
# PENYIMPANAN HASIL
# ==========================================================

hasil_evaluasi = []
hasil_adf_excel = []
hasil_model_sarima = []
hasil_grid_sarima = []
hasil_screening = []


# ==========================================================
# LOOP SEMUA PROVINSI
# ==========================================================

total_provinsi = len(excel.sheet_names)

for idx_provinsi, nama_provinsi in enumerate(
    excel.sheet_names,
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"PROVINSI {idx_provinsi}/{total_provinsi}")
    print(f"Nama : {nama_provinsi}")
    print("=" * 70)

    df = pd.read_excel(
        file_excel,
        sheet_name=nama_provinsi
    )

    df = df.dropna(
        subset=["KABUPATEN/KOTA"]
    )

    total_kota = len(df)

    print("Jumlah kota :", total_kota)

    # ======================================================
    # LOOP SEMUA KABUPATEN/KOTA
    # ======================================================

    for i in range(total_kota):

        nama_kota = str(
            df.iloc[i]["KABUPATEN/KOTA"]
        ).strip()

        # Reset setiap kota
        best = None
        forecast = None

        p = np.nan
        d = np.nan
        q = np.nan
        P = np.nan
        D = np.nan
        Q = np.nan

        mae = np.nan
        rmse = np.nan
        mape = np.nan
        wape = np.nan

        print("\n")
        print("-" * 70)
        print(f"KOTA {i + 1}/{total_kota}")
        print(f"Nama : {nama_kota}")
        print("-" * 70)

        # ==================================================
        # DATA CURAH HUJAN
        # ==================================================

        kolom_tanggal = df.columns[1:]

        tanggal = pd.to_datetime(
            kolom_tanggal,
            errors="coerce"
        )

        data = pd.to_numeric(
            df.iloc[i, 1:].values,
            errors="coerce"
        )

        ts = pd.Series(
            data,
            index=tanggal,
            name="curah_hujan"
        )

        # Buang kolom dengan tanggal tidak valid
        ts = ts[~ts.index.isna()]

        if ts.isna().any():
            print(
                f"Ada {ts.isna().sum()} data kosong. "
                "Kota dilewati."
            )
            continue

        if (ts < 0).any():
            print(
                "Ada nilai curah hujan negatif. "
                "Kota dilewati."
            )
            continue

        if ts.nunique() <= 1:
            print("Data konstan. Kota dilewati.")
            continue

        # ==================================================
        # INFORMASI DATA
        # ==================================================

        print("\nINFORMASI DATA")
        print("-" * 40)
        print("Jumlah Data      :", len(ts))
        print(
            "Tanggal Awal     :",
            ts.index[0].strftime("%Y-%m")
        )
        print(
            "Tanggal Akhir    :",
            ts.index[-1].strftime("%Y-%m")
        )
        print("Missing Value    :", ts.isna().sum())

        # ==================================================
        # TRAIN TEST SPLIT
        # ==================================================

        train = ts.loc[
            "2024-06-01":"2025-11-01"
        ]

        test = ts.loc[
            "2025-12-01":"2026-05-01"
        ]

        print("\nTRAIN TEST SPLIT")
        print("-" * 40)
        print("Jumlah Train :", len(train))
        print("Jumlah Test  :", len(test))

        if len(train) != 18 or len(test) != 6:
            print("Data tidak valid, dilewati.")
            continue

        # ==================================================
        # VALIDATION SPLIT
        # ==================================================

        validation_size = 6

        validation_train = train.iloc[
            :-validation_size
        ].copy()

        validation_actual = train.iloc[
            -validation_size:
        ].copy()

        # ==================================================
        # STATISTIK DESKRIPTIF
        # ==================================================

        print("\nSTATISTIK DESKRIPTIF")
        print("-" * 40)
        print(ts.describe())

        print("\n5 DATA PERTAMA")
        print(ts.head())

        print("\n5 DATA TERAKHIR")
        print(ts.tail())

        # ==================================================
        # ADF TEST
        # ==================================================

        print("\nADF TEST")
        print("-" * 40)

        try:
            hasil_adf = adfuller(train)

            adf_stat = hasil_adf[0]
            p_value = hasil_adf[1]

            print(
                f"ADF Statistic : {adf_stat:.6f}"
            )
            print(
                f"p-value       : {p_value:.6f}"
            )

            print("\nCritical Values")

            for key, value in hasil_adf[4].items():
                print(
                    f"{key} : {value:.6f}"
                )

            if p_value < 0.05:
                status = "Stasioner"
                print(
                    "\nKeputusan : Data STASIONER"
                )
            else:
                status = "Tidak Stasioner"
                print(
                    "\nKeputusan : Data TIDAK STASIONER"
                )

            hasil_adf_excel.append({
                "Provinsi": nama_provinsi,
                "Kabupaten/Kota": nama_kota,
                "ADF Statistic": adf_stat,
                "p-value": p_value,
                "Critical Value 1%": hasil_adf[4]["1%"],
                "Critical Value 5%": hasil_adf[4]["5%"],
                "Critical Value 10%": hasil_adf[4]["10%"],
                "Status": status
            })

        except Exception as e:
            print("ADF Test gagal:", e)

        # ==================================================
        # FOLDER PROVINSI
        # ==================================================

        folder_provinsi = os.path.join(
            folder_output,
            nama_provinsi.replace(" ", "_")
        )

        os.makedirs(
            folder_provinsi,
            exist_ok=True
        )

        nama_file = (
            nama_kota
            .replace(".", "")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )

        # ==================================================
        # ACF
        # ==================================================

        try:
            plt.figure(figsize=(8, 4))

            plot_acf(
                train,
                lags=min(
                    8,
                    len(train) // 2 - 1
                )
            )

            plt.title(
                f"ACF - {nama_kota}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    folder_provinsi,
                    f"{nama_file}_ACF.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        except Exception as e:
            print("ACF gagal:", e)
            plt.close("all")

        # ==================================================
        # PACF
        # ==================================================

        try:
            plt.figure(figsize=(8, 4))

            plot_pacf(
                train,
                lags=min(
                    8,
                    len(train) // 2 - 1
                ),
                method="ywm"
            )

            plt.title(
                f"PACF - {nama_kota}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    folder_provinsi,
                    f"{nama_file}_PACF.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        except Exception as e:
            print("PACF gagal:", e)
            plt.close("all")

        print(
            "Grafik ACF & PACF berhasil disimpan."
        )

        # ==================================================
        # GRID SEARCH SARIMA + LJUNG-BOX + VALIDATION
        # ==================================================

        print("\n")
        print("=" * 60)
        print("GRID SEARCH SARIMA")
        print("=" * 60)

        print(
            "Kriteria pemilihan model:"
        )
        print(
            "1. Model harus lolos "
            "uji Ljung-Box (p-value > 0.05)"
        )
        print(
            "2. Model yang lolos diuji "
            "menggunakan validation 6 bulan"
        )
        print(
            "3. Model dipilih berdasarkan "
            "Validation MAE terkecil"
        )
        print(
            "4. Validation RMSE digunakan "
            "sebagai pembanding"
        )
        print(
            "5. AIC digunakan sebagai "
            "kriteria berikutnya jika diperlukan"
        )
        print("-" * 60)

        hasil_model = []

        # p,d,q = 0-1
        # P,Q = 0-1
        # D = 0
        for p, d, q, P, D, Q in itertools.product(
            range(2),
            range(2),
            range(2),
            [0, 1],
            [0],
            [0, 1]
        ):

            validation_mae = np.nan
            validation_rmse = np.nan
            validation_mape = np.nan
            validation_wape = np.nan
            status_validation = "Tidak Dicek"

            try:
                # ---------------------------------------------
                # FIT MODEL SARIMA PADA TRAIN 18 BULAN
                # ---------------------------------------------

                model = SARIMAX(
                    train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, 12),
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )

                model_fit = model.fit(
                    disp=False
                )

                # ---------------------------------------------
                # AIC
                # ---------------------------------------------

                aic = model_fit.aic

                # ---------------------------------------------
                # RESIDUAL
                # ---------------------------------------------

                residual = model_fit.resid.dropna()

                # ---------------------------------------------
                # UJI LJUNG-BOX
                # ---------------------------------------------

                lag_lb = min(
                    6,
                    len(residual) - 1
                )

                model_df = (
                    p + q + P + Q
                )

                lb_stat = np.nan
                lb_pvalue = np.nan
                status_lb = (
                    "Derajat bebas tidak cukup"
                )

                if lag_lb > model_df:
                    lb_result = acorr_ljungbox(
                        residual,
                        lags=[lag_lb],
                        model_df=model_df,
                        return_df=True
                    )

                    lb_stat = (
                        lb_result["lb_stat"].iloc[0]
                    )

                    lb_pvalue = (
                        lb_result["lb_pvalue"].iloc[0]
                    )

                    # =================================================
                    # VALIDATION HANYA JIKA LJUNG-BOX LOLOS
                    # =================================================

                    if lb_pvalue > 0.05:

                        status_lb = "Lolos"

                        try:
                            # -----------------------------------------
                            # VALIDATION 6 BULAN
                            # -----------------------------------------

                            model_validation = SARIMAX(
                                validation_train,
                                order=(p, d, q),
                                seasonal_order=(
                                    P, D, Q, 12
                                ),
                                enforce_stationarity=True,
                                enforce_invertibility=True
                            )

                            fit_validation = (
                                model_validation.fit(
                                    disp=False
                                )
                            )

                            forecast_validation = (
                                fit_validation.forecast(
                                    steps=validation_size
                                )
                            )

                            forecast_validation = pd.Series(
                                np.maximum(
                                    forecast_validation,
                                    0
                                ),
                                index=validation_actual.index
                            )

                            (
                                validation_mae,
                                validation_rmse,
                                validation_mape,
                                validation_wape
                            ) = hitung_metrik(
                                validation_actual,
                                forecast_validation
                            )

                            status_validation = "Lolos"

                        except Exception as e:
                            status_validation = "Error"

                            print(
                                "VALIDATION ERROR | "
                                f"SARIMA({p},{d},{q})"
                                f"({P},{D},{Q},12) | "
                                f"{e}"
                            )

                    else:
                        status_lb = "Gagal"

                # ---------------------------------------------
                # SIMPAN HASIL GRID SEARCH
                # ---------------------------------------------

                hasil_model.append([
                    p,
                    d,
                    q,
                    P,
                    D,
                    Q,
                    aic,
                    lb_stat,
                    lb_pvalue,
                    status_lb,
                    validation_mae,
                    validation_rmse,
                    validation_mape,
                    validation_wape,
                    status_validation
                ])

                print(
                    f"SARIMA({p},{d},{q})"
                    f"({P},{D},{Q},12)"
                    f" | AIC={aic:.4f}"
                    f" | LB p-value={lb_pvalue:.4f}"
                    f" | Val MAE={validation_mae:.4f}"
                    f" | {status_lb}"
                    f" | Validation={status_validation}"
                )

            except Exception as e:

                print(
                    f"SARIMA({p},{d},{q})"
                    f"({P},{D},{Q},12) gagal : "
                    f"{e}"
                )

        # ==================================================
        # HASIL GRID SEARCH
        # ==================================================

        if len(hasil_model) == 0:

            print(
                "Tidak ada model SARIMA yang berhasil."
            )

            best = None

            hasil_model_sarima.append({
                "Provinsi": nama_provinsi,
                "Kabupaten/Kota": nama_kota,
                "Model SARIMA": "Tidak ada model",
                "p": np.nan,
                "d": np.nan,
                "q": np.nan,
                "P": np.nan,
                "D": np.nan,
                "Q": np.nan,
                "AIC": np.nan,
                "LB Statistic": np.nan,
                "LB p-value": np.nan,
                "Status LB": "Tidak ada model",
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan,
                "WAPE": np.nan,
                "Validation MAE": np.nan,
                "Validation RMSE": np.nan,
                "Validation MAPE": np.nan,
                "Validation WAPE": np.nan,
                "Status Validation": "Tidak Dicek"
            })

            continue

        else:

            hasil_model = pd.DataFrame(
                hasil_model,
                columns=[
                    "p",
                    "d",
                    "q",
                    "P",
                    "D",
                    "Q",
                    "AIC",
                    "LB Statistic",
                    "LB p-value",
                    "Status LB",
                    "Validation MAE",
                    "Validation RMSE",
                    "Validation MAPE",
                    "Validation WAPE",
                    "Status Validation"
                ]
            )

        # ==================================================
        # TAMPILKAN GRID SEARCH
        # ==================================================

        print("\n")
        print("=" * 80)
        print("HASIL GRID SEARCH SARIMA")
        print("=" * 80)

        print(
            hasil_model.to_string(
                index=False
            )
        )

        # ==================================================
        # FILTER MODEL:
        # LJUNG-BOX + VALIDATION
        # ==================================================

        model_lolos = hasil_model[
            (hasil_model["LB p-value"] > 0.05)
            &
            (
                hasil_model["Status Validation"]
                == "Lolos"
            )
        ].copy()

        model_lolos = (
            model_lolos
            .dropna(
                subset=[
                    "Validation MAE",
                    "Validation RMSE"
                ]
            )
            .sort_values(
                by=[
                    "Validation MAE",
                    "Validation RMSE",
                    "AIC"
                ],
                ascending=[
                    True,
                    True,
                    True
                ]
            )
            .reset_index(drop=True)
        )

        print("\n")
        print("=" * 80)
        print(
            "MODEL YANG LOLOS LJUNG-BOX "
            "DAN VALIDATION"
        )
        print("=" * 80)

        if len(model_lolos) == 0:

            print(
                "Tidak ada model yang lolos "
                "Ljung-Box dan validation."
            )

            best = None

        else:

            print(
                model_lolos.to_string(
                    index=False
                )
            )

            # =============================================
            # MODEL TERBAIK
            # =============================================

            best = model_lolos.iloc[0]

            print("\n")
            print("=" * 60)
            print("MODEL SARIMA TERPILIH")
            print("=" * 60)

            print(
                f"SARIMA("
                f"{int(best['p'])},"
                f"{int(best['d'])},"
                f"{int(best['q'])})"
                f"("
                f"{int(best['P'])},"
                f"{int(best['D'])},"
                f"{int(best['Q'])},12)"
            )

            print(
                f"AIC = {best['AIC']:.4f}"
            )
            print(
                f"LB Statistic = "
                f"{best['LB Statistic']:.4f}"
            )
            print(
                f"LB p-value = "
                f"{best['LB p-value']:.4f}"
            )
            print(
                f"Status LB = "
                f"{best['Status LB']}"
            )
            print(
                f"Validation MAE = "
                f"{best['Validation MAE']:.4f}"
            )
            print(
                f"Validation RMSE = "
                f"{best['Validation RMSE']:.4f}"
            )
            print(
                f"Validation MAPE = "
                f"{best['Validation MAPE']:.2f}%"
            )
            print(
                f"Validation WAPE = "
                f"{best['Validation WAPE']:.2f}%"
            )

        # ==================================================
        # SIMPAN GRID SEARCH KE EXCEL
        # ==================================================

        grid_excel = hasil_model.copy()

        grid_excel.insert(
            0,
            "Provinsi",
            nama_provinsi
        )

        grid_excel.insert(
            1,
            "Kabupaten/Kota",
            nama_kota
        )

        grid_excel["Model"] = (
            "SARIMA("
            + grid_excel["p"].astype(int).astype(str)
            + ","
            + grid_excel["d"].astype(int).astype(str)
            + ","
            + grid_excel["q"].astype(int).astype(str)
            + ")("
            + grid_excel["P"].astype(int).astype(str)
            + ","
            + grid_excel["D"].astype(int).astype(str)
            + ","
            + grid_excel["Q"].astype(int).astype(str)
            + ",12)"
        )

        grid_excel = grid_excel[
            [
                "Provinsi",
                "Kabupaten/Kota",
                "Model",
                "AIC",
                "LB Statistic",
                "LB p-value",
                "Status LB",
                "Validation MAE",
                "Validation RMSE",
                "Validation MAPE",
                "Validation WAPE",
                "Status Validation"
            ]
        ]

        hasil_grid_sarima.append(
            grid_excel
        )

        # ==================================================
        # FORECAST SARIMA
        # ==================================================

        if best is not None:

            print()
            print("=" * 60)
            print("FORECAST SARIMA")
            print("=" * 60)

            p = int(best["p"])
            d = int(best["d"])
            q = int(best["q"])

            P = int(best["P"])
            D = int(best["D"])
            Q = int(best["Q"])

            try:

                # Fit ulang pada seluruh 18 bulan training
                model_final = SARIMAX(
                    train,
                    order=(p, d, q),
                    seasonal_order=(
                        P, D, Q, 12
                    ),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                fit_final = model_final.fit(
                    disp=False
                )

                forecast = fit_final.forecast(
                    steps=len(test)
                )

                forecast = pd.Series(
                    np.maximum(
                        forecast,
                        0
                    ),
                    index=test.index
                )

                print("\nForecast")
                print(forecast)

            except Exception as e:

                print(
                    "Forecast gagal :",
                    e
                )

                forecast = pd.Series(
                    [np.nan] * len(test),
                    index=test.index
                )

        else:

            print()
            print("=" * 60)
            print(
                "TIDAK ADA MODEL SARIMA "
                "YANG LOLOS"
            )
            print("=" * 60)

            forecast = pd.Series(
                [np.nan] * len(test),
                index=test.index
            )

            p = np.nan
            d = np.nan
            q = np.nan
            P = np.nan
            D = np.nan
            Q = np.nan

        # ==================================================
        # EVALUASI TEST
        # ==================================================

        print()
        print("=" * 60)
        print("EVALUASI")
        print("=" * 60)

        try:

            mae, rmse, mape, wape = hitung_metrik(
                test,
                forecast
            )

        except Exception as e:

            print(
                "Evaluasi gagal:",
                e
            )

            mae = np.nan
            rmse = np.nan
            mape = np.nan
            wape = np.nan

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAPE : {mape:.2f}%")
        print(f"WAPE : {wape:.2f}%")

        # ==================================================
        # LJUNG-BOX MODEL FINAL
        # ==================================================

        lb_stat_final = np.nan
        lb_pvalue_final = np.nan
        status_lb_final = (
            "Tidak dilakukan"
        )

        if best is not None and forecast.notna().all():

            try:

                residual_final = (
                    fit_final.resid.dropna()
                )

                lag_lb_final = min(
                    6,
                    len(residual_final) - 1
                )

                model_df_final = (
                    p + q + P + Q
                )

                if lag_lb_final > model_df_final:

                    lb_final = acorr_ljungbox(
                        residual_final,
                        lags=[lag_lb_final],
                        model_df=model_df_final,
                        return_df=True
                    )

                    lb_stat_final = (
                        lb_final["lb_stat"].iloc[0]
                    )

                    lb_pvalue_final = (
                        lb_final["lb_pvalue"].iloc[0]
                    )

                    if lb_pvalue_final > 0.05:
                        status_lb_final = "Lolos"
                    else:
                        status_lb_final = "Gagal"

                    print(
                        "\nUJI LJUNG-BOX MODEL FINAL"
                    )
                    print(
                        f"LB Statistic : "
                        f"{lb_stat_final:.4f}"
                    )
                    print(
                        f"p-value      : "
                        f"{lb_pvalue_final:.4f}"
                    )
                    print(
                        f"Status       : "
                        f"{status_lb_final}"
                    )

                else:

                    print(
                        "\nUJI LJUNG-BOX MODEL FINAL"
                    )
                    print(
                        "Tidak dilakukan: "
                        "derajat bebas tidak cukup."
                    )

            except Exception as e:

                print(
                    "Ljung-Box model final gagal:",
                    e
                )

        # ==================================================
        # SIMPAN HASIL MODEL SARIMA TERBAIK
        # ==================================================

        if best is not None:

            hasil_model_sarima.append({

                "Provinsi":
                    nama_provinsi,

                "Kabupaten/Kota":
                    nama_kota,

                "Model SARIMA":
                    (
                        f"SARIMA({p},{d},{q})"
                        f"({P},{D},{Q},12)"
                    ),

                "p": p,
                "d": d,
                "q": q,
                "P": P,
                "D": D,
                "Q": Q,

                "AIC":
                    best["AIC"],

                "LB Statistic":
                    best["LB Statistic"],

                "LB p-value":
                    best["LB p-value"],

                "Status LB":
                    best["Status LB"],

                "Validation MAE":
                    best["Validation MAE"],

                "Validation RMSE":
                    best["Validation RMSE"],

                "Validation MAPE":
                    best["Validation MAPE"],

                "Validation WAPE":
                    best["Validation WAPE"],

                "Status Validation":
                    best["Status Validation"],

                "LB Final Statistic":
                    lb_stat_final,

                "LB Final p-value":
                    lb_pvalue_final,

                "Status LB Final":
                    status_lb_final,

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "MAPE":
                    mape,

                "WAPE":
                    wape
            })

        else:

            hasil_model_sarima.append({

                "Provinsi":
                    nama_provinsi,

                "Kabupaten/Kota":
                    nama_kota,

                "Model SARIMA":
                    "Tidak ada model lolos",

                "p": np.nan,
                "d": np.nan,
                "q": np.nan,
                "P": np.nan,
                "D": np.nan,
                "Q": np.nan,

                "AIC": np.nan,

                "LB Statistic": np.nan,
                "LB p-value": np.nan,
                "Status LB":
                    "Tidak ada model lolos",

                "Validation MAE": np.nan,
                "Validation RMSE": np.nan,
                "Validation MAPE": np.nan,
                "Validation WAPE": np.nan,
                "Status Validation":
                    "Tidak Dicek",

                "LB Final Statistic": np.nan,
                "LB Final p-value": np.nan,
                "Status LB Final":
                    "Tidak dilakukan",

                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan,
                "WAPE": np.nan
            })

        # ==================================================
        # GRAFIK + SCREENING
        # ==================================================

        if (
            best is not None
            and forecast.notna().all()
        ):

            folder_forecast = os.path.join(
                folder_output,
                nama_provinsi.replace(
                    " ", "_"
                ),
                "FORECAST_SARIMA"
            )

            os.makedirs(
                folder_forecast,
                exist_ok=True
            )

            # ------------------------------------------------
            # DATA SCREENING
            # ------------------------------------------------

            actual = np.asarray(
                test.values,
                dtype=float
            )

            pred = np.asarray(
                forecast.values,
                dtype=float
            )

            if (
                len(actual) >= 2
                and np.std(actual) > 0
                and np.std(pred) > 0
            ):

                korelasi = np.corrcoef(
                    actual,
                    pred
                )[0, 1]

            else:

                korelasi = 0.0

            actual_diff = np.diff(actual)
            pred_diff = np.diff(pred)

            valid_direction = (
                (actual_diff != 0)
                &
                (pred_diff != 0)
            )

            if np.sum(valid_direction) > 0:

                actual_direction = np.sign(
                    actual_diff[
                        valid_direction
                    ]
                )

                pred_direction = np.sign(
                    pred_diff[
                        valid_direction
                    ]
                )

                direction_accuracy = (
                    np.mean(
                        actual_direction
                        == pred_direction
                    )
                    * 100
                )

            else:

                direction_accuracy = 0.0

            correlation_score = (
                (korelasi + 1) / 2
            ) * 100

            visual_score = (
                0.60 * correlation_score
                +
                0.40 * direction_accuracy
            )

            print(
                f"[SCREENING SARIMA] "
                f"{nama_kota} | "
                f"Korelasi={korelasi:.3f} | "
                f"Direction="
                f"{direction_accuracy:.2f}% | "
                f"Visual Score="
                f"{visual_score:.2f}"
            )

            # ------------------------------------------------
            # GRAFIK
            # ------------------------------------------------

            plt.figure(
                figsize=(12, 6)
            )

            plt.plot(
                train.index,
                train.values,
                marker="o",
                label="Data Training"
            )

            plt.plot(
                test.index,
                test.values,
                marker="o",
                label="Data Aktual"
            )

            plt.plot(
                forecast.index,
                forecast.values,
                marker="o",
                linestyle="--",
                label="Forecast SARIMA"
            )

            plt.axvline(
                x=test.index[0],
                linestyle=":"
            )

            plt.title(
                f"Forecast Curah Hujan SARIMA - "
                f"{nama_kota}"
                f"\nMAE={mae:.2f} | "
                f"RMSE={rmse:.2f} | "
                f"MAPE={mape:.2f}% | "
                f"WAPE={wape:.2f}%"
            )

            plt.xlabel("Waktu")
            plt.ylabel("Curah Hujan")
            plt.legend()

            plt.grid(
                True,
                alpha=0.3
            )

            plt.gca().xaxis.set_major_formatter(
                mdates.DateFormatter(
                    "%b-%Y"
                )
            )

            plt.xticks(
                rotation=45
            )

            plt.tight_layout()

            nama_file_grafik = (
                f"{nama_provinsi}_"
                f"{nama_kota}_SARIMA.png"
            )

            nama_file_grafik = (
                nama_file_grafik
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )

            plt.savefig(
                os.path.join(
                    folder_forecast,
                    nama_file_grafik
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            # ------------------------------------------------
            # SIMPAN SCREENING
            # ------------------------------------------------

            hasil_screening.append({

                "PROVINSI":
                    nama_provinsi,

                "KABUPATEN_KOTA":
                    nama_kota,

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "MAPE":
                    mape,

                "WAPE":
                    wape,

                "KORELASI":
                    korelasi,

                "DIRECTION_ACCURACY":
                    direction_accuracy,

                "VISUAL_SCORE":
                    visual_score
            })

            # ------------------------------------------------
            # SIMPAN EVALUASI
            # ------------------------------------------------

            hasil_evaluasi.append({

                "Provinsi":
                    nama_provinsi,

                "Kota":
                    nama_kota,

                "Model":
                    (
                        f"SARIMA({p},{d},{q})"
                        f"({P},{D},{Q},12)"
                    ),

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "MAPE":
                    mape,

                "WAPE":
                    wape
            })


# ==========================================================
# SIMPAN HASIL SCREENING SARIMA
# ==========================================================

df_screening = pd.DataFrame(
    hasil_screening
)

if not df_screening.empty:

    df_screening = df_screening.sort_values(
        by="VISUAL_SCORE",
        ascending=False
    ).reset_index(drop=True)

    file_screening = os.path.join(
        folder_output,
        "SCREENING_GRAFIK_SARIMA_BEFORE.xlsx"
    )

    df_screening.to_excel(
        file_screening,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("TOP 20 GRAFIK SARIMA BEFORE")
    print("=" * 80)

    print(
        df_screening[
            [
                "PROVINSI",
                "KABUPATEN_KOTA",
                "WAPE",
                "KORELASI",
                "DIRECTION_ACCURACY",
                "VISUAL_SCORE"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\nFile screening berhasil dibuat:"
    )
    print(file_screening)

else:

    print(
        "\nTidak ada grafik SARIMA yang "
        "dapat masuk screening."
    )


# ==========================================================
# SIMPAN EVALUASI
# ==========================================================

evaluasi_df = pd.DataFrame(
    hasil_evaluasi
)

evaluasi_df.to_excel(
    "EVALUASI_SARIMA.xlsx",
    index=False
)


# ==========================================================
# SIMPAN ADF
# ==========================================================

df_adf = pd.DataFrame(
    hasil_adf_excel
)

df_adf.to_excel(
    "HASIL_ADF_SARIMA.xlsx",
    index=False
)


# ==========================================================
# SIMPAN MODEL TERBAIK
# ==========================================================

df_model = pd.DataFrame(
    hasil_model_sarima
)

df_model.to_excel(
    "HASIL_MODEL_SARIMA.xlsx",
    index=False
)


# ==========================================================
# SIMPAN GRID SEARCH
# ==========================================================

if len(hasil_grid_sarima) > 0:

    grid_df = pd.concat(
        hasil_grid_sarima,
        ignore_index=True
    )

    grid_df.to_excel(
        "GRID_SEARCH_SARIMA.xlsx",
        index=False
    )

    print(
        "GRID_SEARCH_SARIMA.xlsx "
        "berhasil disimpan."
    )

else:

    print(
        "GRID_SEARCH_SARIMA.xlsx "
        "tidak dibuat karena tidak ada "
        "hasil grid search."
    )


print("\n")
print("=" * 60)
print("HASIL SARIMA BERHASIL DISIMPAN")
print("=" * 60)
print("File : HASIL_ADF_SARIMA.xlsx")
print("File : GRID_SEARCH_SARIMA.xlsx")
print("File : HASIL_MODEL_SARIMA.xlsx")
print("File : EVALUASI_SARIMA.xlsx")
print(
    "File : SCREENING_GRAFIK_SARIMA_BEFORE.xlsx"
)
