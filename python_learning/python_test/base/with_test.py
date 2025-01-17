class DatabaseConnection:

    def __enter__(self):
        # 模拟数据库连接
        print("Connecting to the database...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 模拟关闭数据库连接
        print("Closing the database connection...")
        if exc_type is not None:
            print(f"An exception occurred: {exc_val}")

    def query(self, sql):
        # 模拟执行SQL查询
        print(f"Executing SQL: {sql}")


# 使用自定义的DatabaseConnection类
with DatabaseConnection() as db:
    db.query("SELECT * FROM users")
