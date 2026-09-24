"""Generate a small SYNTHETIC dataset with the same schema as the private raw data.

The rows are fabricated from a simple price model (depreciation with age/km, brand tier,
power); no real listing is reproduced. Used by the tests and for smoke-running the pipeline.

    python scripts/make_sample.py
"""
import numpy as np
import pandas as pd

from car_price.config import ROOT

rng = np.random.default_rng(17)
N = 1500

BRANDS = {  # name: (base price factor, models)
    "FIAT": (0.7, ["Panda", "500", "Tipo"]), "VOLKSWAGEN": (1.0, ["Golf", "Polo", "Passat"]),
    "FORD": (0.9, ["Fiesta", "Focus"]), "RENAULT": (0.8, ["Clio", "Megane"]),
    "OPEL": (0.8, ["Corsa", "Astra"]), "PEUGEOT": (0.85, ["208", "308"]),
    "BMW": (1.6, ["Serie 1", "Serie 3", "X1"]), "AUDI": (1.6, ["A3", "A4", "Q3"]),
    "MERCEDES-BENZ": (1.7, ["Classe A", "Classe C"]), "PORSCHE": (3.5, ["Cayenne", "Macan"]),
    "FERRARI": (9.0, ["California", "F430"]), "DR": (0.9, ["DR5"]),
}
names = list(BRANDS)
weights = np.array([12, 10, 8, 7, 6, 6, 9, 8, 8, 1.5, 0.3, 0.5])
brand = rng.choice(names, N, p=weights / weights.sum())
model = [rng.choice(BRANDS[b][1]) for b in brand]
factor = np.array([BRANDS[b][0] for b in brand])

year = rng.integers(2003, 2020, N)
age = 2019 - year
km = np.clip(age * rng.normal(14000, 5000, N), 500, None).round()
cv = np.clip(rng.normal(95, 30, N) * np.sqrt(factor), 40, 600).round()
fuel = rng.choice(["Diesel", "Benzina", "GPL", "Metano", "Elettrico"], N, p=[.6, .28, .07, .03, .02])
gear = rng.choice(["Cambio manuale", "Cambio automatico"], N, p=[.65, .35])

price = (9000 * factor * np.exp(-0.09 * age) * (cv / 95) ** 0.7 * np.exp(-2e-6 * km)
         * np.where(gear == "Cambio automatico", 1.1, 1.0) * np.exp(rng.normal(0, 0.12, N)))

df = pd.DataFrame({
    "price": price.round().astype(int).clip(500, None),
    "MatriculationMonth": rng.choice(["Gennaio", "Marzo", "Giugno", "No Mese"], N),
    "MatriculationYear": year, "km": km, "cv": cv, "FuelType": fuel, "gearboxType": gear,
    "Consume": np.nan, "city": "Roma", "province": rng.choice(["Roma", "Milano", "Torino"], N),
    "Brand": brand, "Model": model,
    "Preparation": [f"{m} 1.{rng.integers(2, 9)} {rng.choice(['Sport', 'Business', 'Comfort', 'Base'])} 5p."
                    for m in model],
    "Engine": (cv * rng.normal(14, 1.5, N)).round(), "Seats": rng.choice([4, 5, 5, 5, 7], N),
    "ConsumeFuel": rng.normal(5.5, 1.2, N).round(1), "ConsumeFuelNotUrban": np.nan,
    "Emissions": rng.normal(125, 25, N).round(), "Color": rng.choice(["Nero", "Bianco", "Grigio"], N),
    "Metallizzato": np.nan,
    "Airbag": rng.choice(["Airbag anteriori e laterali", "Airbag conducente", np.nan], N),
    "NumberDoors": rng.choice(["4 o 5 porte", "2 o 3 porte", np.nan], N),
    "EmissionClass": rng.choice(["Euro 6", "Euro 5", "Euro 4", np.nan], N),
    "AirConditioning": rng.choice(["Climatizzatore manuale", "Climatizzatore automatico", np.nan], N),
    "Interior": np.nan,
})
# a few realistic defects so the cleaning rules are exercised
df.loc[rng.choice(N, 15, replace=False), "Engine"] = np.nan
df.loc[rng.choice(N, 5, replace=False), "ConsumeFuel"] = 6000
df.loc[rng.choice(N, 5, replace=False), "Seats"] = np.nan

out = ROOT / "data" / "sample" / "auto_price_sample.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding="utf-8")
print(f"wrote {out} {df.shape}")
