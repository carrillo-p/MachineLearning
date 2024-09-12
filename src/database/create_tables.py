from connection import create_connection, close_connection

def create_tables():
    connection = create_connection()
    cursor = connection.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        xgboost_prediction INT,
        xgboost_probability FLOAT,
        logistic_prediction INT,
        logistic_probability FLOAT,
        stacked_prediction INT,               
        stacked_probability FLOAT, 
        neural_prediction INT,  
        neural_probability FLOAT,             
        gender VARCHAR(10),
        customer_type VARCHAR(20),
        age INT,
        travel_type VARCHAR(20),
        flight_class VARCHAR(10),
        flight_distance FLOAT,
        inflight_wifi INT,
        departure_convenience INT,
        online_booking INT,
        gate_location INT,
        food_drink INT,
        online_boarding INT,
        seat_comfort INT,
        inflight_entertainment INT,
        onboard_service INT,
        legroom_service INT,
        baggage_handling INT,
        checkin_service INT,
        inflight_service_personal INT,
        cleanliness INT,
        departure_delay INT,
        arrival_delay INT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INT AUTO_INCREMENT PRIMARY KEY,
        rating INT,
        comment TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Crear la nueva tabla para almacenar toda la información
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS new_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        gender VARCHAR(10),
        customer_type VARCHAR(20),
        age INT,
        travel_type VARCHAR(20),
        flight_class VARCHAR(10),
        flight_distance FLOAT,
        inflight_wifi INT,
        departure_convenience INT,
        online_booking INT,
        gate_location INT,
        food_drink INT,
        online_boarding INT,
        seat_comfort INT,
        inflight_entertainment INT,
        onboard_service INT,
        legroom_service INT,
        baggage_handling INT,
        checkin_service INT,
        inflight_service_personal INT,
        cleanliness INT,
        departure_delay INT,
        arrival_delay INT,
        xgboost_prediction INT,
        xgboost_probability FLOAT,
        logistic_prediction INT,
        logistic_probability FLOAT,
        stacked_prediction INT,               
        stacked_probability FLOAT, 
        neural_prediction INT,  
        neural_probability FLOAT,             
        feedback_model1 INT,
        feedback_model2 INT,
        feedback_model3 INT,
        feedback_model4 INT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    connection.commit()
    close_connection(connection)

if __name__ == "__main__":
    create_tables()