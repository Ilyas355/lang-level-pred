# src/constants.py
TARGET_ACC = 0.75
TARGET_F1_MACRO = 0.70

CEFR2ID = {"A1":0,"A2":1,"B1":2,"B2":3,"C1":4,"C2":5}
ID2CEFR = {v:k for k,v in CEFR2ID.items()}

RAW_SCORE_COLUMNS = {
    "speaking_score","reading_score","listening_score","writing_score","overall_score"
}
ENGINEERED_OK_SUFFIXES = ("_minus_avg", "_level")
ENGINEERED_OK_EXACT = {
    "productive_dominant","strongest_skill","weakest_skill","second_weakest_skill",
    "strength_weakness_gap","learning_profile"
}
