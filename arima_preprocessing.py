# ==========================================================
# ARIMA DENGAN PREPROCESSING
# PREPROCESSING = LOG TRANSFORMATION (log1p)
# ==========================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use("seaborn-v0_8-whitegrid")

import os
import warnings
import itertools

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

def hitung_metrik(actual, prediksi):

    actual = pd.Series(actual, dtype=float)
    prediksi = pd.Series(
        prediksi,
        index=actual.index,
        dtype=float
    )

    valid = actual.notna() & prediksi.notna()

    actual = actual[valid]
    prediksi = prediksi[valid]

    mae = mean_absolute_error(
        actual,
        prediksi
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediksi
        )
    )

    mask_mape = actual != 0

    if mask_mape.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    actual[mask_mape]
                    - prediksi[mask_mape]
                )
                / actual[mask_mape]
            )
        ) * 100

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

file_excel = "DATASET ARIMA SETELAH PREPROCESSING.xlsm"

excel = pd.ExcelFile(file_excel)


# ==========================================================
# FOLDER HASIL GRAFIK
# ==========================================================

folder_output = "HASIL_GRAFIK_PREPROCESSING"

os.makedirs(
    folder_output,
    exist_ok=True
)


# ==========================================================
# INFORMASI SHEET
# ==========================================================

print("Jumlah sheet:", len(excel.sheet_names))
print()

for i, sheet in enumerate(
    excel.sheet_names,
    start=1
):

    print(i, sheet)

print("\nSELESAI")


# ==========================================================
# MENYIMPAN HASIL
# ==========================================================

hasil_evaluasi = []

hasil_adf_excel = []

hasil_grid_arima = []

hasil_model_arima = []

hasil_screening = []
# ==========================================================
# LOOP SEMUA PROVINSI
# ==========================================================

total_provinsi = len(
    excel.sheet_names
)


for idx_provinsi, nama_provinsi in enumerate(
    excel.sheet_names,
    start=1
):

    print("\n")
    print("=" * 70)
    print(
        f"PROVINSI {idx_provinsi}/{total_provinsi}"
    )
    print(
        f"Nama : {nama_provinsi}"
    )
    print("=" * 70)


    print("\n" + "=" * 60)

    print(
        "PROVINSI :",
        nama_provinsi
    )


    # ======================================================
    # BACA SHEET
    # ======================================================

    df = pd.read_excel(
        file_excel,
        sheet_name=nama_provinsi
    )


    # ======================================================
    # HAPUS BARIS TANPA NAMA KABUPATEN/KOTA
    # ======================================================

    df = df.dropna(
        subset=["KABUPATEN/KOTA"]
    )


    print(
        "Jumlah kota :",
        len(df)
    )


    total_kota = len(df)


    # ======================================================
    # LOOP SEMUA KABUPATEN/KOTA
    # ======================================================

    for i in range(total_kota):


        nama_kota = str(
            df.iloc[i]["KABUPATEN/KOTA"]
        ).strip()


        print("\n")
        print("-" * 70)

        print(
            f"KOTA {i+1}/{total_kota}"
        )

        print(
            f"Nama : {nama_kota}"
        )

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

        ts = ts[~ts.index.isna()]


        # ==================================================
        # INFORMASI DATA ASLI
        # ==================================================

        print("\nINFORMASI DATA ASLI")
        print("-" * 40)

        print(
            "Jumlah Data      :",
            len(ts)
        )

        print(
            "Tanggal Awal     :",
            ts.index[0].strftime("%Y-%m")
        )

        print(
            "Tanggal Akhir    :",
            ts.index[-1].strftime("%Y-%m")
        )

        print(
            "Missing Value    :",
            ts.isna().sum()
        )


        # ==================================================
        # CEK MISSING VALUE
        # ==================================================

        if ts.isna().any():

            print(
                "Terdapat missing value. "
                "Kota dilewati."
            )

            continue


        # ==================================================
        # ==================================================
        # PREPROCESSING
        # LOG TRANSFORMATION
        # ==================================================
        # ==================================================

        print("\n")
        print("=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)

        print(
            "Metode preprocessing : Log Transformation"
        )

        print(
            "Transformasi         : log1p(x)"
        )


        # --------------------------------------------------
        # log1p digunakan karena data curah hujan
        # dapat memiliki nilai 0.
        #
        # log1p(x) = log(1+x)
        #
        # sehingga:
        # log1p(0) = 0
        # --------------------------------------------------

        ts_preprocessed = np.log1p(ts)


        print("\n5 DATA ASLI")
        print(
            ts.head()
        )


        print("\n5 DATA SETELAH PREPROCESSING")
        print(
            ts_preprocessed.head()
        )


        # ==================================================
        # TRAIN TEST SPLIT
        # ==================================================

        train = ts.loc["2024-06-01":"2025-11-01"]
        test = ts.loc["2025-12-01":"2026-05-01"]

        # ==================================================
        # TRAIN TEST PREPROCESSING
        # ==================================================

        train_preprocessed = ts_preprocessed.iloc[:18]

        # ==================================================
        # VALIDATION SPLIT
        # ==================================================

        validation_size = 6

        validation_train_preprocessed = (
            train_preprocessed.iloc[:-validation_size].copy()
        )

        validation_actual = (
            train.iloc[-validation_size:].copy()
        )

        print("\nTRAIN TEST SPLIT")
        print("-" * 40)

        print(
            "Jumlah Train :",
            len(train)
        )

        print(
            "Jumlah Test  :",
            len(test)
        )


        if len(train) != 18 or len(test) != 6:

            print(
                "Data tidak valid, dilewati."
            )

            continue


        # ==================================================
        # CEK DATA KONSTAN
        # ==================================================

        if train_preprocessed.nunique() <= 1:

            print(
                "Data konstan."
            )

            print(
                "ADF, ACF, PACF, dan ARIMA dilewati."
            )

            continue


        # ==================================================
        # STATISTIK DESKRIPTIF DATA ASLI
        # ==================================================

        print("\nSTATISTIK DESKRIPTIF DATA ASLI")
        print("-" * 40)

        print(
            ts.describe()
        )


        # ==================================================
        # STATISTIK DESKRIPTIF DATA PREPROCESSING
        # ==================================================

        print("\nSTATISTIK DESKRIPTIF SETELAH PREPROCESSING")
        print("-" * 40)

        print(
            ts_preprocessed.describe()
        )


        # ==================================================
        # ADF TEST
        # DILAKUKAN PADA DATA HASIL PREPROCESSING
        # ==================================================

        print("\nADF TEST")
        print("-" * 40)


        try:

            hasil_adf = adfuller(
                train_preprocessed
            )


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

                "Provinsi":
                    nama_provinsi,

                "Kabupaten/Kota":
                    nama_kota,

                "ADF Statistic":
                    adf_stat,

                "p-value":
                    p_value,

                "Critical Value 1%":
                    hasil_adf[4]["1%"],

                "Critical Value 5%":
                    hasil_adf[4]["5%"],

                "Critical Value 10%":
                    hasil_adf[4]["10%"],

                "Status":
                    status

            })


        except Exception as e:

            print(
                "ADF Test gagal:",
                e
            )


        # ==================================================
        # FOLDER PROVINSI
        # ==================================================

        folder_provinsi = os.path.join(

            folder_output,

            nama_provinsi.replace(
                " ",
                "_"
            )

        )


        os.makedirs(
            folder_provinsi,
            exist_ok=True
        )


        nama_file = (

            nama_kota
            .replace(".", "")
            .replace("/", "_")
            .replace(" ", "_")

        )


        # ==================================================
        # ACF
        # MENGGUNAKAN DATA HASIL PREPROCESSING
        # ==================================================

        plt.figure(
            figsize=(8, 4)
        )


        plot_acf(

            train_preprocessed,

            lags=min(
                8,
                len(train_preprocessed)//2 - 1
            )

        )


        plt.title(
            f"ACF Preprocessing - {nama_kota}"
        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                folder_provinsi,

                f"{nama_file}_ACF_PREPROCESSING.png"

            )

        )


        plt.close("all")


        # ==================================================
        # PACF
        # MENGGUNAKAN DATA HASIL PREPROCESSING
        # ==================================================

        plt.figure(
            figsize=(8, 4)
        )


        plot_pacf(

            train_preprocessed,

            lags=min(
                8,
                len(train_preprocessed)//2 - 1
            ),

            method="ywm"

        )


        plt.title(
            f"PACF Preprocessing - {nama_kota}"
        )


        plt.tight_layout()


        plt.savefig(

            os.path.join(

                folder_provinsi,

                f"{nama_file}_PACF_PREPROCESSING.png"

            )

        )


        plt.close("all")


        print(
            "Grafik ACF & PACF preprocessing berhasil disimpan."
        )


        # ==================================================
        # GRID SEARCH ARIMA
        # ==================================================

        print("\n")
        print("=" * 60)
        print("GRID SEARCH ARIMA")
        print("=" * 60)


        print(
            "Kriteria pemilihan model:"
        )

        print(
            "1. Model harus lolos uji Ljung-Box "
            "(p-value > 0.05)"
        )

        print(
            "2. Model yang lolos diuji menggunakan "
            "validation 6 bulan"
        )

        print(
            "3. Model dipilih berdasarkan Validation MAE "
            "terkecil"
        )

        print(
            "4. Jika diperlukan, Validation RMSE digunakan "
            "sebagai pembanding"
        )

        print(
            "5. AIC digunakan sebagai kriteria berikutnya "
            "jika nilai validation sama"
        )

        print("-" * 60)


        hasil_model = []


        # ==================================================
        # GRID SEARCH
        # ==================================================

        for p, d, q in itertools.product(

            range(3),

            range(2),

            range(3)

        ):

            validation_mae = np.nan
            validation_rmse = np.nan
            validation_mape = np.nan
            validation_wape = np.nan
            status_validation = "Tidak Dicek"


            try:


                # ------------------------------------------
                # FIT ARIMA PADA DATA PREPROCESSING
                # ------------------------------------------

                model = ARIMA(

                    train_preprocessed,

                    order=(p, d, q)

                )


                model_fit = model.fit()


                # ------------------------------------------
                # AIC
                # ------------------------------------------

                aic = model_fit.aic


                # ------------------------------------------
                # RESIDUAL
                # ------------------------------------------

                residual = (
                    model_fit.resid
                    .dropna()
                )


                # ------------------------------------------
                # LJUNG-BOX
                # ------------------------------------------

                lag_lb = min(
                    6,
                    len(residual) - 1
                )

                model_df = p + q

                lb_stat = np.nan
                lb_pvalue = np.nan
                status_lb = "Derajat bebas tidak cukup"

                if lag_lb > model_df:

                    lb_result = acorr_ljungbox(
                        residual,
                        lags=[lag_lb],
                        model_df=model_df,
                        return_df=True
                    )

                    lb_stat = lb_result["lb_stat"].iloc[0]
                    lb_pvalue = lb_result["lb_pvalue"].iloc[0]

                    if lb_pvalue > 0.05:
                        status_lb = "Lolos"
                    else:
                        status_lb = "Gagal"

                    if status_lb == "Lolos":

                        try:

                            model_validation = ARIMA(
                                validation_train_preprocessed,
                                order=(p, d, q)
                            )

                            fit_validation = model_validation.fit()

                            forecast_validation_preprocessed = (
                                fit_validation.forecast(
                                    steps=validation_size
                                )
                            )

                            forecast_validation = pd.Series(
                                np.expm1(
                                    forecast_validation_preprocessed
                                ),
                                index=validation_actual.index
                            )

                            forecast_validation = forecast_validation.clip(
                                lower=0
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

                        except Exception:
                            status_validation = "Error"

                # ------------------------------------------
                # SIMPAN HASIL
                # ------------------------------------------

                hasil_model.append([

                    p,

                    d,

                    q,

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

                    f"ARIMA({p},{d},{q})"
                    f" | AIC = {aic:.4f}"
                    f" | LB p-value = {lb_pvalue:.4f}"
                    f" | {status_lb}"

                )


            except Exception as e:

                print(

                    f"ARIMA({p},{d},{q}) gagal : {e}"

                )


        # ==================================================
        # HASIL GRID SEARCH
        # ==================================================

        if len(hasil_model) == 0:

            print(
                "Tidak ada model yang berhasil."
            )

            continue


        else:


            hasil_model = pd.DataFrame(

                hasil_model,

                columns=[

                    "p",

                    "d",

                    "q",

                    "AIC",

                    "LB Statistic",

                    "LB p-value",

                    "Status Ljung-Box",

                    "Validation MAE",

                    "Validation RMSE",

                    "Validation MAPE",

                    "Validation WAPE",

                    "Status Validation"

                ]

            )


            # ==================================================
            # TAMPILKAN SEMUA MODEL
            # ==================================================

            print("\n")
            print("=" * 70)
            print("HASIL GRID SEARCH ARIMA")
            print("=" * 70)


            print(
                hasil_model.to_string(
                    index=False
                )
            )


            # ==================================================
            # FILTER MODEL LOLOS LJUNG-BOX
            # ==================================================

            model_lolos = hasil_model[
                (hasil_model["LB p-value"] > 0.05) &
                (hasil_model["Status Validation"] == "Lolos")
            ].copy()


            print("\n")
            print("=" * 70)
            print("MODEL YANG LOLOS UJI LJUNG-BOX")
            print("=" * 70)


            if len(model_lolos) == 0:

                print(
                    "Tidak ada model yang lolos "
                    "Ljung-Box dan validation."
                )

                best = None

            else:

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

                best = model_lolos.iloc[0]
                # ==================================================
                # MODEL TERBAIK
                # ==================================================
                print("\n")
                print("=" * 60)
                print("MODEL ARIMA TERPILIH")
                print("=" * 60)


                print(

                    f"ARIMA("
                    f"{int(best['p'])},"
                    f"{int(best['d'])},"
                    f"{int(best['q'])})"

                )


                print(
                    f"AIC          = "
                    f"{best['AIC']:.4f}"
                )


                print(
                    f"LB Statistic = "
                    f"{best['LB Statistic']:.4f}"
                )


                print(
                    f"LB p-value   = "
                    f"{best['LB p-value']:.4f}"
                )


                print(
                    f"Status       = "
                    f"{best['Status Ljung-Box']}"
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
            # SIMPAN GRID SEARCH
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

                "ARIMA("

                + grid_excel["p"]
                    .astype(int)
                    .astype(str)

                + ","

                + grid_excel["d"]
                    .astype(int)
                    .astype(str)

                + ","

                + grid_excel["q"]
                    .astype(int)
                    .astype(str)

                + ")"

            )


            grid_excel = grid_excel[

                [

                    "Provinsi",

                    "Kabupaten/Kota",

                    "Model",

                    "AIC",

                    "LB Statistic",

                    "LB p-value",

                    "Status Ljung-Box",

                    "Validation MAE",

                    "Validation RMSE",

                    "Validation MAPE",

                    "Validation WAPE",

                    "Status Validation"

                ]

            ]


            hasil_grid_arima.append(
                grid_excel
            )


        # ==================================================
        # FORECAST
        # ==================================================

        if best is not None:


            print()
            print("=" * 60)
            print("FORECAST ARIMA")
            print("=" * 60)


            p = int(
                best["p"]
            )

            d = int(
                best["d"]
            )

            q = int(
                best["q"]
            )


            try:


                # ------------------------------------------
                # MODEL FINAL
                # ------------------------------------------

                model_final = ARIMA(

                    train_preprocessed,

                    order=(p, d, q)

                )


                fit_final = (
                    model_final.fit()
                )


                # ------------------------------------------
                # FORECAST PADA SKALA LOG
                # ------------------------------------------

                forecast_preprocessed = (
                    fit_final.forecast(
                        steps=len(test)
                    )
                )


                print(
                    "\nForecast hasil preprocessing:"
                )

                print(
                    forecast_preprocessed
                )


                # ------------------------------------------
                # KEMBALIKAN KE SKALA ASLI
                # ------------------------------------------

                forecast = pd.Series(

                    np.expm1(
                        forecast_preprocessed
                    ),

                    index=test.index

                )


                # ------------------------------------------
                # PASTIKAN TIDAK ADA NILAI NEGATIF
                # ------------------------------------------

                forecast = forecast.clip(
                    lower=0
                )


                print(
                    "\nForecast setelah dikembalikan "
                    "ke skala asli:"
                )

                print(
                    forecast
                )


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
            print("TIDAK ADA MODEL ARIMA YANG LOLOS")
            print("=" * 60)


            forecast = pd.Series(

                [np.nan] * len(test),

                index=test.index

            )


            p = np.nan
            d = np.nan
            q = np.nan


        # ==================================================
        # EVALUASI
        # MENGGUNAKAN DATA ASLI
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

            print("Evaluasi gagal:", e)

            mae = np.nan
            rmse = np.nan
            mape = np.nan
            wape = np.nan


        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"MAPE : {mape:.2f}%"
        )

        print(
            f"WAPE : {wape:.2f}%"
        )
        # ==================================================
        # SIMPAN HASIL MODEL TERBAIK
        # ==================================================

        if best is not None:


            hasil_model_arima.append({

                "Provinsi":
                    nama_provinsi,

                "Kabupaten/Kota":
                    nama_kota,

                "Model ARIMA":
                    f"ARIMA({p},{d},{q})",

                "p":
                    p,

                "d":
                    d,

                "q":
                    q,

                "AIC":
                    best["AIC"],

                "LB Statistic":
                    best["LB Statistic"],

                "LB p-value":
                    best["LB p-value"],

                "Status Ljung-Box":
                    best["Status Ljung-Box"],

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "MAPE":
                    mape,
                
                 "WAPE":
                    wape

            })


            # ==================================================
            # GRAFIK FORECAST ARIMA DENGAN PREPROCESSING
            # + SCREENING VISUAL
            # ==================================================

            # --------------------------------------------------
            # SCREENING: DATA AKTUAL DAN FORECAST
            # --------------------------------------------------

            actual = np.asarray(test.values, dtype=float)
            pred = np.asarray(forecast.values, dtype=float)

            # --------------------------------------------------
            # KORELASI AKTUAL VS FORECAST
            # --------------------------------------------------

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

            # --------------------------------------------------
            # DIRECTION ACCURACY
            # --------------------------------------------------

            actual_diff = np.diff(actual)
            pred_diff = np.diff(pred)

            mask = (
                (actual_diff != 0)
                &
                (pred_diff != 0)
            )

            if np.sum(mask) > 0:

                direction_accuracy = (
                    np.mean(
                        np.sign(actual_diff[mask])
                        ==
                        np.sign(pred_diff[mask])
                    )
                    * 100
                )

            else:

                direction_accuracy = 0.0


            # --------------------------------------------------
            # KORELASI DIUBAH KE SKALA 0-100
            # --------------------------------------------------

            correlation_score = (
                (korelasi + 1) / 2
            ) * 100


            # --------------------------------------------------
            # VISUAL SCORE
            # --------------------------------------------------
            # 60% kesesuaian pola
            # 40% kesesuaian arah perubahan
            # --------------------------------------------------

            visual_score = (
                0.60 * correlation_score
                +
                0.40 * direction_accuracy
            )


            # --------------------------------------------------
            # PRINT SCREENING
            # --------------------------------------------------

            print(
                f"[SCREENING ARIMA PREPROCESSING] "
                f"{nama_kota} | "
                f"WAPE={wape:.2f}% | "
                f"Korelasi={korelasi:.3f} | "
                f"Direction={direction_accuracy:.2f}% | "
                f"Visual Score={visual_score:.2f}"
            )


            # ==================================================
            # FOLDER OUTPUT GRAFIK
            # ==================================================

            folder_forecast = os.path.join(
                folder_output,
                nama_provinsi.replace(" ", "_"),
                "FORECAST"
            )

            os.makedirs(
                folder_forecast,
                exist_ok=True
            )


            # ==================================================
            # GRAFIK
            # ==================================================

            plt.figure(
                figsize=(12, 6)
            )


            # --------------------------------------------------
            # DATA TRAINING
            # --------------------------------------------------

            plt.plot(
                train.index,
                train.values,
                marker="o",
                label="Data Training"
            )


            # --------------------------------------------------
            # DATA AKTUAL
            # --------------------------------------------------

            plt.plot(
                test.index,
                test.values,
                marker="o",
                label="Data Aktual"
            )


            # --------------------------------------------------
            # FORECAST ARIMA
            # --------------------------------------------------

            plt.plot(
                forecast.index,
                forecast.values,
                marker="o",
                linestyle="--",
                label="Forecast ARIMA"
            )


            # --------------------------------------------------
            # GARIS PEMISAH TRAINING DAN TESTING
            # --------------------------------------------------

            plt.axvline(
                x=test.index[0],
                linestyle=":"
            )


            # --------------------------------------------------
            # JUDUL + METRIK
            # --------------------------------------------------

            plt.title(
                f"Forecast Curah Hujan ARIMA Dengan Preprocessing - {nama_kota}"
                f"\nMAE={mae:.2f} | "
                f"RMSE={rmse:.2f} | "
                f"MAPE={mape:.2f}% | "
                f"WAPE={wape:.2f}%"
            )


            # --------------------------------------------------
            # LABEL SUMBU
            # --------------------------------------------------

            plt.xlabel(
                "Waktu"
            )

            plt.ylabel(
                "Curah Hujan"
            )


            # --------------------------------------------------
            # LEGEND
            # --------------------------------------------------

            plt.legend()


            # --------------------------------------------------
            # GRID
            # --------------------------------------------------

            plt.grid(
                True,
                alpha=0.3
            )


            # --------------------------------------------------
            # FORMAT TANGGAL
            # --------------------------------------------------

            plt.gca().xaxis.set_major_formatter(
                mdates.DateFormatter("%b-%Y")
            )

            plt.xticks(
                rotation=45
            )


            # --------------------------------------------------
            # LAYOUT
            # --------------------------------------------------

            plt.tight_layout()


            # --------------------------------------------------
            # NAMA FILE
            # --------------------------------------------------

            nama_file_grafik = (
                f"{nama_provinsi}_{nama_kota}_ARIMA_PREPROCESSING.png"
            )

            nama_file_grafik = (
                nama_file_grafik
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )


            # --------------------------------------------------
            # SIMPAN PNG
            # --------------------------------------------------

            plt.savefig(
                os.path.join(
                    folder_forecast,
                    nama_file_grafik
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


            # ==================================================
            # SIMPAN HASIL SCREENING
            # ==================================================

            hasil_screening.append({

                "PROVINSI": nama_provinsi,

                "KABUPATEN_KOTA": nama_kota,

                "MAE": mae,

                "RMSE": rmse,

                "MAPE": mape,

                "WAPE": wape,

                "KORELASI": korelasi,

                "DIRECTION_ACCURACY": direction_accuracy,

                "VISUAL_SCORE": visual_score

            })

            # ==================================================
            # SIMPAN EVALUASI
            # ==================================================

            hasil_evaluasi.append({

                "Provinsi":
                    nama_provinsi,

                "Kota":
                    nama_kota,

                "Model":
                    f"ARIMA({p},{d},{q})",

                "Preprocessing":
                    "Log Transformation (log1p)",

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


            hasil_model_arima.append({

                "Provinsi":
                    nama_provinsi,

                "Kabupaten/Kota":
                    nama_kota,

                "Model ARIMA":
                    "Tidak ada model lolos",

                "p":
                    np.nan,

                "d":
                    np.nan,

                "q":
                    np.nan,

                "AIC":
                    np.nan,

                "LB Statistic":
                    np.nan,

                "LB p-value":
                    np.nan,

                "Status Ljung-Box":
                    "Tidak ada model lolos",

                "MAE":
                    np.nan,

                "RMSE":
                    np.nan,

                "MAPE":
                    np.nan,

                "WAPE":
                    np.nan

            })

# ==========================================================
# HASIL SCREENING GRAFIK ARIMA PREPROCESSING
# ==========================================================

if len(hasil_screening) > 0:

    df_screening = pd.DataFrame(
        hasil_screening
    )

    # Urutkan berdasarkan Visual Score tertinggi
    df_screening = df_screening.sort_values(
        by="VISUAL_SCORE",
        ascending=False
    )

    # ------------------------------------------------------
    # SIMPAN KE EXCEL
    # ------------------------------------------------------

    file_screening = os.path.join(
        folder_output,
        "SCREENING_GRAFIK_ARIMA_PREPROCESSING.xlsx"
    )

    df_screening.to_excel(
        file_screening,
        index=False
    )

    # ------------------------------------------------------
    # TAMPILKAN 20 TERATAS
    # ------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TOP 20 SCREENING GRAFIK ARIMA PREPROCESSING")
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
        ].head(20).to_string(index=False)
    )

    print("\n")
    print(
        f"File screening berhasil disimpan:\n"
        f"{file_screening}"
    )

else:

    print(
        "\nTidak ada data screening yang berhasil dikumpulkan."
    )
# ==========================================================
# SIMPAN HASIL KE EXCEL
# ==========================================================


# ==========================================================
# HASIL EVALUASI
# ==========================================================

evaluasi_df = pd.DataFrame(
    hasil_evaluasi
)


evaluasi_df.to_excel(

    "EVALUASI_ARIMA_PREPROCESSING.xlsx",

    index=False

)


print(
    "\nEVALUASI_ARIMA_PREPROCESSING.xlsx berhasil disimpan."
)


# ==========================================================
# HASIL ADF
# ==========================================================

df_adf = pd.DataFrame(
    hasil_adf_excel
)


df_adf.to_excel(

    "HASIL_ADF_ARIMA_PREPROCESSING.xlsx",

    index=False

)


print(
    "HASIL_ADF_ARIMA_PREPROCESSING.xlsx berhasil disimpan."
)


# ==========================================================
# HASIL GRID SEARCH
# ==========================================================

if len(hasil_grid_arima) > 0:


    grid_df = pd.concat(

        hasil_grid_arima,

        ignore_index=True

    )


    grid_df.to_excel(

        "GRID_SEARCH_ARIMA_PREPROCESSING.xlsx",

        index=False

    )


    print(
        "GRID_SEARCH_ARIMA_PREPROCESSING.xlsx berhasil disimpan."
    )


else:


    print(

        "GRID_SEARCH_ARIMA_PREPROCESSING.xlsx "
        "tidak dibuat karena tidak ada hasil grid search."

    )


# ==========================================================
# HASIL MODEL ARIMA TERBAIK
# ==========================================================

df_model_arima = pd.DataFrame(
    hasil_model_arima
)


df_model_arima.to_excel(

    "HASIL_MODEL_ARIMA_PREPROCESSING.xlsx",

    index=False

)


print(
    "HASIL_MODEL_ARIMA_PREPROCESSING.xlsx berhasil disimpan."
)


# ==========================================================
# SELESAI
# ==========================================================

print("\n")
print("=" * 60)
print("SEMUA HASIL ARIMA PREPROCESSING BERHASIL DISIMPAN")
print("=" * 60)

print(
    "File : HASIL_ADF_ARIMA_PREPROCESSING.xlsx"
)

print(
    "File : GRID_SEARCH_ARIMA_PREPROCESSING.xlsx"
)

print(
    "File : HASIL_MODEL_ARIMA_PREPROCESSING.xlsx"
)

print(
    "File : EVALUASI_ARIMA_PREPROCESSING.xlsx"
)

print(
    "\nPreprocessing yang digunakan:"
)

print(
    "Log Transformation menggunakan log1p(x)"
)

print(
    "Forecast dikembalikan ke skala asli menggunakan expm1(x)"
)

import sys
import io
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter


# ==========================================================
# FUNGSI BANTU STYLING
# ==========================================================

def style_header(cell, bg="1F4E79", fg="FFFFFF", bold=True, size=11):
    cell.font = Font(bold=bold, color=fg, size=size, name="Calibri")
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def style_subheader(cell, bg="2E75B6", fg="FFFFFF"):
    cell.font = Font(bold=True, color=fg, size=10, name="Calibri")
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

def style_data(cell, center=False, bold=False, bg=None):
    cell.font = Font(size=10, name="Calibri", bold=bold)
    cell.alignment = Alignment(
        horizontal="center" if center else "left",
        vertical="center", wrap_text=True
    )
    if bg:
        cell.fill = PatternFill("solid", fgColor=bg)

def thin_border():
    s = Side(style="thin", color="CCCCCC")
    return Border(left=s, right=s, top=s, bottom=s)

def apply_border(ws, row, col):
    ws.cell(row=row, column=col).border = thin_border()

def set_col_width(ws, col, width):
    ws.column_dimensions[get_column_letter(col)].width = width

def kategori_mape(mape):
    try:
        m = float(mape)
        if m < 10:   return "Sangat Baik"
        elif m < 20: return "Baik"
        elif m < 50: return "Layak"
        else:        return "Buruk"
    except:
        return "-"

def warna_kategori(kategori):
    return {
        "Sangat Baik": "C6EFCE",
        "Baik":        "FFEB9C",
        "Layak":       "FFCC99",
        "Buruk":       "FFC7CE",
    }.get(kategori, "FFFFFF")


# ==========================================================
# BUAT WORKBOOK UTAMA
# ==========================================================

wb = Workbook()


# ==========================================================
# SHEET 1: RINGKASAN EVALUASI
# ==========================================================

ws1 = wb.active
ws1.title = "Ringkasan Evaluasi"

# Judul
ws1.merge_cells("A1:I1")
ws1["A1"] = "RINGKASAN EVALUASI MODEL ARIMA DENGAN PREPROCESSING (Log1p)"
style_header(ws1["A1"], bg="1F4E79", size=13)
ws1.row_dimensions[1].height = 30

ws1.merge_cells("A2:I2")
ws1["A2"] = "Data Curah Hujan Nasional | Train: 18 bulan | Test: 6 bulan"
ws1["A2"].font = Font(italic=True, size=10, color="595959")
ws1["A2"].alignment = Alignment(horizontal="center")
ws1.row_dimensions[2].height = 18

# Header kolom
headers = ["No", "Provinsi", "Kabupaten/Kota", "Model ARIMA",
           "MAE", "RMSE", "MAPE (%)", "Kategori MAPE", "Kategori MAPE", "Preprocessing"]
for col, h in enumerate(headers, 1):
    c = ws1.cell(row=4, column=col, value=h)
    style_subheader(c)
    apply_border(ws1, 4, col)
ws1.row_dimensions[4].height = 22

# Lebar kolom
widths = [
    5, 20, 25, 18, 10,
    10, 10, 10, 16, 25
]
for col, w in enumerate(widths, 1):
    set_col_width(ws1, col, w)

# Data dari hasil_evaluasi
for i, row in enumerate(hasil_evaluasi, 1):
    r = i + 4
    kat = kategori_mape(row.get("MAPE", ""))
    warna = warna_kategori(kat)

    vals = [
        i,
        row.get("Provinsi", ""),
        row.get("Kota", ""),
        row.get("Model", ""),
        round(row.get("MAE", 0), 4),
        round(row.get("RMSE", 0), 4),
        round(row.get("MAPE", 0), 2),
        round(row.get("WAPE", 0), 2),
        kat,
        row.get(
            "Preprocessing",
            "Log Transformation (log1p)"
        ),
    ]

    bg_row = "F2F7FF" if i % 2 == 0 else "FFFFFF"

    for col, val in enumerate(vals, 1):
        c = ws1.cell(row=r, column=col, value=val)
        center = col in [1, 4, 5, 6, 7, 8]
        bg = warna if col == 8 else bg_row
        style_data(c, center=center, bg=bg)
        apply_border(ws1, r, col)

    ws1.row_dimensions[r].height = 16

# Freeze header
ws1.freeze_panes = "A5"


# ==========================================================
# SHEET 2: STATISTIK RINGKASAN
# ==========================================================

import numpy as np

ws2 = wb.create_sheet("Statistik Ringkasan")

ws2.merge_cells("A1:E1")
ws2["A1"] = "STATISTIK RINGKASAN MODEL ARIMA PREPROCESSING"
style_header(ws2["A1"], bg="1F4E79", size=13)
ws2.row_dimensions[1].height = 30

# Ambil data numerik
maes  = [r["MAE"]  for r in hasil_evaluasi if r.get("MAE")  is not None and not (isinstance(r["MAE"],  float) and np.isnan(r["MAE"]))]
rmses = [r["RMSE"] for r in hasil_evaluasi if r.get("RMSE") is not None and not (isinstance(r["RMSE"], float) and np.isnan(r["RMSE"]))]
mapes = [r["MAPE"] for r in hasil_evaluasi if r.get("MAPE") is not None and not (isinstance(r["MAPE"], float) and np.isnan(r["MAPE"]))]
wapes = [r["WAPE"] for r in hasil_evaluasi if r.get("WAPE") is not None and not (isinstance(r["WAPE"], float) and np.isnan(r["WAPE"]))]

stat_headers = ["Statistik", "MAE", "RMSE", "MAPE (%)", "WAPE (%)"]
for col, h in enumerate(stat_headers, 1):
    c = ws2.cell(row=3, column=col, value=h)
    style_subheader(c)

stat_rows = [
    (
        "Jumlah Wilayah",
        len(maes),
        len(rmses),
        len(mapes),
        len(wapes)
    ),

    (
        "Mean",
        round(np.mean(maes), 4),
        round(np.mean(rmses), 4),
        round(np.mean(mapes), 2),
        round(np.mean(wapes), 2)
    ),

    (
        "Median",
        round(np.median(maes), 4),
        round(np.median(rmses), 4),
        round(np.median(mapes), 2),
        round(np.median(wapes), 2)
    ),

    (
        "Std Dev",
        round(np.std(maes), 4),
        round(np.std(rmses), 4),
        round(np.std(mapes), 2),
        round(np.std(wapes), 2)
    ),

    (
        "Min",
        round(np.min(maes), 4),
        round(np.min(rmses), 4),
        round(np.min(mapes), 2),
        round(np.min(wapes), 2)
    ),

    (
        "Max",
        round(np.max(maes), 4),
        round(np.max(rmses), 4),
        round(np.max(mapes), 2),
        round(np.max(wapes), 2)
    ),
]

for i, row in enumerate(stat_rows, 4):
    for col, val in enumerate(row, 1):
        c = ws2.cell(row=i, column=col, value=val)
        style_data(c, center=(col > 1), bold=(col == 1))
        apply_border(ws2, i, col)

# Distribusi kategori MAPE
ws2.cell(row=12, column=1, value="DISTRIBUSI KATEGORI MAPE")
style_subheader(ws2.cell(row=12, column=1), bg="2E75B6")
ws2.merge_cells("A12:D12")

kat_headers = ["Kategori", "Jumlah Wilayah", "Persentase", "Keterangan MAPE"]
for col, h in enumerate(kat_headers, 1):
    c = ws2.cell(row=13, column=col, value=h)
    style_subheader(c, bg="4472C4")

categories = [
    ("Sangat Baik", lambda m: m < 10,        "MAPE < 10%"),
    ("Baik",        lambda m: 10 <= m < 20,   "10% ≤ MAPE < 20%"),
    ("Layak",       lambda m: 20 <= m < 50,   "20% ≤ MAPE < 50%"),
    ("Buruk",       lambda m: m >= 50,         "MAPE ≥ 50%"),
]

for i, (label, cond, keterangan) in enumerate(categories, 14):
    count = sum(1 for m in mapes if cond(m))
    pct = round(count / len(mapes) * 100, 1) if mapes else 0
    warna = warna_kategori(label)
    vals = [label, count, f"{pct}%", keterangan]
    for col, val in enumerate(vals, 1):
        c = ws2.cell(row=i, column=col, value=val)
        style_data(c, center=(col > 1), bg=warna if col == 1 else None)
        apply_border(ws2, i, col)

for col, w in enumerate([22, 16, 14, 22], 1):
    set_col_width(ws2, col, w)


# ==========================================================
# SHEET 3: HASIL ADF TEST
# ==========================================================

ws3 = wb.create_sheet("Uji Stasioneritas (ADF)")

ws3.merge_cells("A1:H1")
ws3["A1"] = "HASIL UJI STASIONERITAS (AUGMENTED DICKEY-FULLER TEST)"
style_header(ws3["A1"], bg="1F4E79", size=13)
ws3.row_dimensions[1].height = 30

adf_headers = ["No", "Provinsi", "Kabupaten/Kota", "ADF Statistic",
               "p-value", "Critical 1%", "Critical 5%", "Status"]
for col, h in enumerate(adf_headers, 1):
    c = ws3.cell(row=3, column=col, value=h)
    style_subheader(c)

for i, row in enumerate(hasil_adf_excel, 1):
    r = i + 3
    status = row.get("Status", "")
    bg_status = "C6EFCE" if status == "Stasioner" else "FFC7CE"
    bg_row = "F2F7FF" if i % 2 == 0 else "FFFFFF"

    vals = [
        i,
        row.get("Provinsi", ""),
        row.get("Kabupaten/Kota", ""),
        round(row.get("ADF Statistic", 0), 6),
        round(row.get("p-value", 0), 6),
        round(row.get("Critical Value 1%", 0), 6),
        round(row.get("Critical Value 5%", 0), 6),
        status,
    ]
    for col, val in enumerate(vals, 1):
        c = ws3.cell(row=r, column=col, value=val)
        bg = bg_status if col == 8 else bg_row
        style_data(c, center=(col in [1, 4, 5, 6, 7, 8]), bg=bg)
        apply_border(ws3, r, col)
    ws3.row_dimensions[r].height = 15

ws3.freeze_panes = "A4"
for col, w in enumerate([5, 20, 25, 16, 12, 14, 14, 16], 1):
    set_col_width(ws3, col, w)


# ==========================================================
# SHEET 4: DETAIL MODEL ARIMA TERBAIK
# ==========================================================

ws4 = wb.create_sheet("Model ARIMA Terbaik")

ws4.merge_cells("A1:K1")
ws4["A1"] = "DETAIL MODEL ARIMA TERBAIK PER KABUPATEN/KOTA"
style_header(ws4["A1"], bg="1F4E79", size=13)
ws4.row_dimensions[1].height = 30

model_headers = ["No", "Provinsi", "Kabupaten/Kota", "Model ARIMA",
                 "p", "d", "q", "AIC", "LB Statistic", "LB p-value", "Status LB"]
for col, h in enumerate(model_headers, 1):
    c = ws4.cell(row=3, column=col, value=h)
    style_subheader(c)

for i, row in enumerate(hasil_model_arima, 1):
    r = i + 3
    bg_row = "F2F7FF" if i % 2 == 0 else "FFFFFF"
    status_lb = row.get("Status Ljung-Box", "")
    bg_lb = "C6EFCE" if status_lb == "Lolos" else "FFC7CE"

    vals = [
        i,
        row.get("Provinsi", ""),
        row.get("Kabupaten/Kota", ""),
        row.get("Model ARIMA", ""),
        row.get("p", ""),
        row.get("d", ""),
        row.get("q", ""),
        round(float(row.get("AIC", 0)), 4) if row.get("AIC") else "",
        round(float(row.get("LB Statistic", 0)), 4) if row.get("LB Statistic") else "",
        round(float(row.get("LB p-value", 0)), 4) if row.get("LB p-value") else "",
        status_lb,
    ]
    for col, val in enumerate(vals, 1):
        c = ws4.cell(row=r, column=col, value=val)
        bg = bg_lb if col == 11 else bg_row
        style_data(c, center=(col in [1, 4, 5, 6, 7, 8, 9, 10, 11]), bg=bg)
        apply_border(ws4, r, col)
    ws4.row_dimensions[r].height = 15

ws4.freeze_panes = "A4"
for col, w in enumerate([5, 20, 25, 18, 5, 5, 5, 12, 14, 12, 14], 1):
    set_col_width(ws4, col, w)


# ==========================================================
# SHEET 5: RANKING WILAYAH
# ==========================================================

ws5 = wb.create_sheet("Ranking Wilayah")

ws5.merge_cells("A1:H1")
ws5["A1"] = "RANKING WILAYAH BERDASARKAN MAPE (TERBAIK → TERBURUK)"
style_header(ws5["A1"], bg="1F4E79", size=13)
ws5.row_dimensions[1].height = 30

rank_headers = ["Rank", "Provinsi", "Kabupaten/Kota", "Model ARIMA",
                "MAE", "RMSE", "MAPE (%)", "Kategori"]
for col, h in enumerate(rank_headers, 1):
    c = ws5.cell(row=3, column=col, value=h)
    style_subheader(c)

sorted_eval = sorted(
    [r for r in hasil_evaluasi if r.get("MAPE") is not None and not (isinstance(r["MAPE"], float) and np.isnan(r["MAPE"]))],
    key=lambda x: x["MAPE"]
)

for i, row in enumerate(sorted_eval, 1):
    r = i + 3
    kat = kategori_mape(row["MAPE"])
    warna = warna_kategori(kat)
    bg_row = "F2F7FF" if i % 2 == 0 else "FFFFFF"

    vals = [
        i,
        row.get("Provinsi", ""),
        row.get("Kota", ""),
        row.get("Model", ""),
        round(row.get("MAE", 0), 4),
        round(row.get("RMSE", 0), 4),
        round(row.get("MAPE", 0), 2),
        kat,
    ]
    for col, val in enumerate(vals, 1):
        c = ws5.cell(row=r, column=col, value=val)
        bg = warna if col == 8 else bg_row
        style_data(c, center=(col in [1, 4, 5, 6, 7, 8]), bg=bg)
        apply_border(ws5, r, col)
    ws5.row_dimensions[r].height = 15

ws5.freeze_panes = "A4"
for col, w in enumerate([6, 20, 25, 18, 10, 10, 10, 16], 1):
    set_col_width(ws5, col, w)


# ==========================================================
# SIMPAN FILE
# ==========================================================

nama_file_output = "LAPORAN_LENGKAP_ARIMA_PREPROCESSING.xlsx"

wb.save(nama_file_output)

print(f"\n{'='*60}")
print(f"FILE LAPORAN BERHASIL DISIMPAN")
print(f"{'='*60}")
print(f"Nama file : {nama_file_output}")
print(f"\nISI FILE:")
print(f"  Sheet 1 : Ringkasan Evaluasi   ({len(hasil_evaluasi)} wilayah)")
print(f"  Sheet 2 : Statistik Ringkasan")
print(f"  Sheet 3 : Uji Stasioneritas ADF ({len(hasil_adf_excel)} wilayah)")
print(f"  Sheet 4 : Detail Model Terbaik  ({len(hasil_model_arima)} wilayah)")
print(f"  Sheet 5 : Ranking Wilayah       ({len(sorted_eval)} wilayah)")
print(f"{'='*60}")
