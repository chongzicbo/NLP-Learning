import pymysql


def get_connection(host: str, username: str, port: int, password: str, database: str):
    conn = pymysql.connect(
        host=host, user=username, port=port, passwd=password, database=database
    )
    return conn


conn = get_connection(
    host="10.10.0.86",
    username="readonlys",
    port=3306,
    password="Iij6AeCo",
    database="mdpipub",
)

cursor = conn.cursor()
sql = "select hash_key,article_author_id from submission_manuscript where id=2489608"
cursor.execute(sql)
for row in cursor.fetchall():
    print(row)
