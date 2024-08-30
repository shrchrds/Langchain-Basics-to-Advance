import sqlite3

# Connect to sqlite
connection =sqlite3.connect("Sales.db")

cursor = connection.cursor()

# Create table

table_info = """
        CREATE TABLE Customer (
            CustomerID INTEGER PRIMARY KEY,
            FirstName TEXT NOT NULL,
            LastName TEXT NOT NULL,
            Email TEXT NOT NULL,
            PhoneNumber TEXT NOT NULL,
            Address TEXT NOT NULL,
            City TEXT NOT NULL,
            State TEXT NOT NULL,
            ZipCode TEXT NOT NULL,
            AccountBalance REAL NOT NULL
        )
"""
cursor.execute(table_info)

# Insert sample records into the Customer table
cursor.execute("""
INSERT INTO Customer (CustomerID, FirstName, LastName, Email, PhoneNumber, Address, City, State, ZipCode, AccountBalance) VALUES
(1, 'John', 'Doe', 'john.doe@example.com', '1234567890', '123 Elm St', 'Springfield', 'IL', '62701', 1500.75),
(2, 'Jane', 'Smith', 'jane.smith@example.com', '2345678901', '456 Oak St', 'Springfield', 'IL', '62702', 2500.50),
(3, 'Alice', 'Johnson', 'alice.johnson@example.com', '3456789012', '789 Pine St', 'Springfield', 'IL', '62703', 3200.00),
(4, 'Bob', 'Brown', 'bob.brown@example.com', '4567890123', '101 Maple St', 'Springfield', 'IL', '62704', 4100.25),
(5, 'Carol', 'Davis', 'carol.davis@example.com', '5678901234', '202 Birch St', 'Springfield', 'IL', '62705', 5300.75),
(6, 'David', 'Wilson', 'david.wilson@example.com', '6789012345', '303 Cedar St', 'Springfield', 'IL', '62706', 6200.50),
(7, 'Emma', 'Moore', 'emma.moore@example.com', '7890123456', '404 Walnut St', 'Springfield', 'IL', '62707', 7100.00),
(8, 'Frank', 'Taylor', 'frank.taylor@example.com', '8901234567', '505 Chestnut St', 'Springfield', 'IL', '62708', 8200.25),
(9, 'Grace', 'Anderson', 'grace.anderson@example.com', '9012345678', '606 Ash St', 'Springfield', 'IL', '62709', 9300.75),
(10, 'Henry', 'Thomas', 'henry.thomas@example.com', '0123456789', '707 Poplar St', 'Springfield', 'IL', '62710', 10400.50);
"""
)

# Display all the records
print("Inserted records are: ")
data =cursor.execute("SELECT * FROM Customer")

for row in data:
    print(row)

# Commit changes in database

connection.commit()
connection.close()