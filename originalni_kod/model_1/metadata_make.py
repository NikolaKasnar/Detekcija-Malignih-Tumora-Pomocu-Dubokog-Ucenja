import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("ISIC_2020_Training_GroundTruth.csv")
df.sort_values(by='image_name', ascending=True, inplace=True)

df['sex_encoded'] = df['sex'].map({'female': 0, 'male': 1})
df['sex_encoded'] = df['sex_encoded'].fillna(0).astype(int)
df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('unknown')
df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median()).astype(np.float32)


# 2. One-hot encode anatom_site_general_challenge
anatom_onehot = pd.get_dummies(df['anatom_site_general_challenge'], prefix='site')

# 3. Combine features
X = pd.concat([df['sex_encoded'], df['age_approx'], anatom_onehot], axis=1)

# 4. Convert to NumPy array
X_array = X.to_numpy()
X_array= np.array( X_array, dtype=np.float32)
# 5. Save
np.save("features.npy", X_array)

# 1. One-hot encode diagnosis
le = LabelEncoder()
labels_index = le.fit_transform(df['diagnosis'])  # shape: [N], values: 0 to 8

# Optional: save class names for decoding later
np.save("diagnosis_classes.npy", le.classes_)

# 2. Save as .npy
np.save("labels_diagnosis.npy", labels_index.astype(np.int64))