import sklearn
from sklearn.linear_model import LogisticRegression

print("sklearn:", sklearn.__version__)

model = LogisticRegression()
print("Has multi_class:", hasattr(model, "multi_class"))
print(model.__dict__)