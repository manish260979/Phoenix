# Databricks notebook source
# MAGIC %md
# MAGIC #Phoenix PNX 845 
# MAGIC # Change Attribute names into Camelcases 
# MAGIC # Transformations applied
# MAGIC 1. Convert resnet_feature_vector from binary to float array
# MAGIC 2. Ignore resnet_feature_vector with length less than 50
# MAGIC 3. Filter required columns
# MAGIC 4. Column transformation on types
# MAGIC 
# MAGIC ### Run  Existing Job Run at 5 AM Everyday 
# MAGIC Note:
# MAGIC All notebook specific packages are installed as first step,
# MAGIC Python interpreter will be restarted after installation

# COMMAND ----------

# Install notebook specific packages
# (can't use both this method and dbutils.library.installPyPi methods in same notebook)
%pip install pymongo
%pip install dnspython
%pip install requests
artifact_token = dbutils.secrets.get(scope="phoenix_secret_scope", key="PHOENIX_ARTIFACTS_SECRET")
%pip install --extra-index-url=https://$artifact_token@pkgs.dev.azure.com/tuesday-development/_packaging/tuesday-pypi/pypi/simple/ phoenix-utils=="0.121.0"


# COMMAND ----------

import os
import pickle
from datetime import date, timedelta
from pymongo import MongoClient
from typing import Optional, List
import pandas as pd

from phoenix_utils import new_relic_event


def decode_binary_feature_with_retries(raw_embedding: bytes) -> Optional[List[float]]:
    return pickle.loads(raw_embedding).astype("float").tolist()


def change_key_name(items):
    if isinstance(items, list):
        try:
            for item in items:
                item["link_type"] = item.pop("linkType", item.get("link_type", str()))
        except Exception:
            pass
    else:
        items = []
    return items


# Mongo connections are in env hardcoded
client = MongoClient(host=dbutils.secrets.get(scope = "phoenix_secret_scope", key = "MONGODB_CONNECTION_ARTICLES"))

# Output file names are defined here
as_of_date = date.today().strftime("%Y/%m/%d")
output_base_path = "/mnt/phoenix_artifacts/centralds/mongo_dumps_dev"
output_path = f"{output_base_path}/{as_of_date}"

previous_day = (date.today() - timedelta(days=1)).strftime("%Y/%m/%d")
previous_day_path = f"{output_base_path}/{previous_day}"

# COMMAND ----------

# Load article collection to a pandas dataframe
articles = client["article-data"]["article"].find({})
df = pd.DataFrame.from_dict(articles)

# COMMAND ----------

# DBTITLE 1,Column  Renaming 
# Developer : Manish Sharma 
# Date : 17/07/2021
# Description : Task performed according to JIRA TICKET PNX 845 .Column Renaming Using CAMEL CASES As PER PNX 845 Requirement .Using this 
#               approach because Amending index may effect Other work 

df.rename(columns = {"_id":"id"}, inplace="True")
df.rename(columns = {"external_product_id":"externalProductId"}, inplace="True")
df.rename(columns = {"image_url":"imageUrl"}, inplace="True")
df.rename(columns = {"product_type":"productType"}, inplace="True")
df.rename(columns = {"description":"description"}, inplace="True")
df.rename(columns = {"lead_article":"leadArticle"}, inplace="True")
df.rename(columns = {"stroke_id":"strokeId"}, inplace="True")
df.rename(columns = {"title":"title"}, inplace="True")
df.rename(columns = {"gender":"gender"}, inplace="True")
df.rename(columns = {"category":"category"}, inplace="True")
df.rename(columns = {"master_category_breadcrumb":"masterCategoryBreadcrumb"}, inplace="True")
df.rename(columns = {"master_category_id":"masterCategoryId"}, inplace="True")
df.rename(columns = {"additional_mages":"additionalImages"}, inplace="True")
df.rename(columns = {"category_breadcrumb":"categoryBreadCrumb"}, inplace="True")
df.rename(columns = {"previous_price":"previousPrice"}, inplace="True")
df.rename(columns = {"stock_level":"stockLevel"}, inplace="True")
df.rename(columns = {"is_active":"isActive"}, inplace="True")
df.rename(columns = {"in_stock":"inStock"}, inplace="True")
df.rename(columns = {"previous_price":"previousPrice"}, inplace="True")
df.rename(columns = {"additional_images":"additionalImages"}, inplace="True")
df.rename(columns = {"dominant_colour":"dominantColour"}, inplace="True")

# COMMAND ----------

# Verify bad data
sp_previous_good_data = None

bad_data = df.loc[df["resnet_feature_vector"].str.len() < 51]
if bad_data.empty:
    print("No bad articles")
else:
    print(f"{len(bad_data.index)} bad articles")

    if len(bad_data.index) > 0:
        # create new relic event counting broken features
        event = [
            {
                "eventType": "PhoenixResnetFeatureVectorErrors",
                "amount": len(bad_data.index)
            }
        ]
        new_relic_event(event)

    print(bad_data["id"])
    bad_list = bad_data["id"].to_list()

    # check if any of the bad ones are available in yesterday's run.
    try:
        last_day_feed = spark.read.parquet(previous_day_path)

        # get good data as spark dataframe. This will be appended before writing file.
        sp_previous_good_data = last_day_feed.filter(last_day_feed._id.isin(bad_list))
    except Exception:
        pass

    # Get good data
    df = df.loc[df["resnet_feature_vector"].str.len() > 50]
print(f"Articles count - {len(df.index)}")

del bad_data

# COMMAND ----------

# DBTITLE 1,Data clean-up - Provide all column transformation here.
# Developer : Manish Sharma 
# Date : 17/07/2021
# Description : Task performed according to JIRA TICKET PNX 845 


# Convert Vector
df["resnet_feature_vector"] = df["resnet_feature_vector"].apply(decode_binary_feature_with_retries)

string_columns = [
    "id",
    "externalProductId",
    "imageUrl",
    "description",
    "leadArticle",
    "productType",
    "strokeId",
    "title",
    "gender",
    "category",
    "masterCategoryBreadcrumb",
    "masterCategoryId",
    "categoryBreadCrumb",
]
double_columns = ["price", "previousPrice", "stockLevel"]
array_columns = [
    "additionalImages",
    "attributes",
    "resnet_feature_vector",  ## Left it because it might be used in other Application with the same Name 
    "dominantColour",
    "linked_articles",    ## Left it because it might be used in other Application with the same Name
]
boolean_columns = [
    "isActive",
    "inStock",
]

df = df[string_columns + double_columns + array_columns + boolean_columns]

df[string_columns] = df[string_columns].astype(str)
df[double_columns] = df[double_columns].astype(float)
df[boolean_columns] = df[boolean_columns].astype(bool)
df["linked_articles"] = df["linked_articles"].apply(change_key_name)

# COMMAND ----------

# pandas Dataframe is converted to spark Dataframe to ensure compatibility for downstream process
print(f"Writing output in {output_path}")
article_feed = spark.createDataFrame(df)

# For bad data in today's run, use yesterdays data if available.
# This could create a data drift when same data is unavailable for long period
if sp_previous_good_data:
    article_feed = article_feed.union(sp_previous_good_data)

# Write df to file
article_feed.write.parquet(output_path, mode="overwrite")