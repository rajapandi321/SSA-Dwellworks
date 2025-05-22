example_prompt_template = [
    {"question" : "What is the conversion rate for Dell for each region YTD?",
      "sql query" : '''
        SELECT 
            GD.Region,
            ROUND((SUM(O.RentedOptyCount * 1.0) / NULLIF(SUM(O.OptyCount * 1.0), 0)) * 100, 0) AS ConversionRate
        FROM CHKPI_OPTY_FACT O
        JOIN CHKPI_CUSTOMER_DIMENSION C ON O.CORP_CLIENT_ID = C.Corp_Client_Id
        JOIN CHKPI_GEO_DIMENSION GD ON O.Geo_ID = GD.Geo_Id
        JOIN CHKPI_DATE_DIMENSION D ON O.OptyCreateDate = D.Date
        WHERE C.CustomerName LIKE 'Dell%'
        AND D.Date >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1) AND D.Date <= GETDATE()
        GROUP BY GD.Region
        ORDER BY ConversionRate DESC;
      '''
    },
    {"question" : "What is the average nightly rate for a 1 bedroom in London?",
    "sql query" : '''
        SELECT ROUND(SUM(MonthlyRentUSD) / SUM(LenghtOfStay), 0) AS AverageNightlyRate
        FROM CHKPI_TENANCY_FACT TF
        JOIN CHKPI_TENANCY_DIMENSION T ON TF.Cur_Tenancy_Id = T.Cur_Tenancy_Id
        JOIN CHKPI_APT_DIMENSION A ON A.Cur_Apt_Id = TF.Cur_Apt_Id
        JOIN CHKPI_GEO_DIMENSION G ON TF.Geo_Id = G.Geo_Id
        WHERE A.Bedrooms = '1 Bedroom'
        AND G.City = 'London'
        AND TF.ActualDate >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1)
        AND TF.ActualDate <= GETDATE();
        '''
    },
    {"question" : "Which regions have the highest number of reservations created by corporate clients YTD?",
     "sql query" : '''
        SELECT 
            GD.Region, 
            COUNT(DISTINCT TF.Cur_Tenancy_Id) AS ReservationsCount
        FROM 
            CHKPI_TENANCY_FACT TF
        JOIN 
            CHKPI_TENANCY_DIMENSION TD ON TF.Cur_Tenancy_Id = TD.Cur_Tenancy_Id
        JOIN 
            CHKPI_GEO_DIMENSION GD ON TF.Geo_Id = GD.Geo_Id
        JOIN 
            CHKPI_CUSTOMER_DIMENSION CD ON TF.Corp_Client_Id = CD.Corp_Client_Id
        JOIN 
            CHKPI_DATE_DIMENSION D ON TF.ActualDate = D.Date
        WHERE 
            D.Date >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1) 
            AND D.Date <= GETDATE()
        GROUP BY 
            GD.Region
        ORDER BY 
            ReservationsCount DESC;
        '''
    },
    {"question" : "What are my top booked locations in EMEA?",
         "sql query" : '''
            SELECT
                    Top 10
                    GD.City,
                    COUNT(DISTINCT TF.Cur_Tenancy_Id) AS NumberOfBookings
                FROM
                    CHKPI_TENANCY_FACT TF
                JOIN
                    CHKPI_GEO_DIMENSION GD ON TF.Geo_Id = GD.Geo_Id
                JOIN
                    CHKPI_DATE_DIMENSION D ON TF.ActualDate = D.Date
                WHERE
                    D.Date >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1)
                    AND D.Date <= GETDATE()
                    AND GD.Region = 'EMEA'
                    AND GD.CITY <> ''
                GROUP BY
                    GD.City
                ORDER BY
                    NumberOfBookings DESC;
            '''
    },
    {
        "question": "Which region has the longest ALOS?",
        "sql query": '''
            SELECT TOP 10
            GD.Region,
            ROUND(SUM(TF.LenghtOfStay * 1.0) / COUNT(DISTINCT TF.Cur_Tenancy_Id), 0) AS AverageLengthOfStay
            FROM CHKPI_TENANCY_FACT TF
            JOIN CHKPI_GEO_DIMENSION GD ON TF.Geo_Id = GD.Geo_Id
            WHERE TF.ActualDate >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1) AND TF.ActualDate <= GETDATE()
            GROUP BY GD.Region
            ORDER BY AverageLengthOfStay DESC;
        '''
    },
    {
        "question": "Please graph the ADR for Nike for the past 6 months?",
        "sql query": '''
            SELECT 
                FORMAT(D.Date, 'yyyy-MM') AS Month,
                ROUND(SUM(TF.MonthlyRentUSD * 1.0) / SUM(TF.LenghtOfStay), 0) AS AverageDailyRate
            FROM CHKPI_TENANCY_FACT TF
            JOIN CHKPI_TENANCY_DIMENSION TD ON TF.Cur_Tenancy_Id = TD.Cur_Tenancy_Id
            JOIN CHKPI_CUSTOMER_DIMENSION CD ON TF.Corp_Client_Id = CD.Corp_Client_Id
            JOIN CHKPI_DATE_DIMENSION D ON TF.ActualDate = D.Date
            WHERE 
                CD.CustomerName LIKE 'Nike%' 
                AND D.Date >= CAST(DATEADD(MONTH, -6, GETDATE()) AS DATE)
                AND D.Date <= CAST(GETDATE() AS DATE)
            GROUP BY FORMAT(D.Date, 'yyyy-MM')
            ORDER BY Month DESC;
        '''
    },
    {
        "question": "Please give me the ADR, ALOS and conversion rate for Google?",
        "sql query": '''
            SELECT 
                -- ADR
                ROUND(SUM(TF.MonthlyRentUSD * 1.0) / NULLIF(SUM(TF.LenghtOfStay), 0), 0) AS AverageDailyRate,
                
                -- ALOS
                ROUND(SUM(TF.LenghtOfStay * 1.0) / NULLIF(COUNT(DISTINCT TF.Cur_Tenancy_Id), 0), 0) AS AverageLengthOfStay,
                
                -- Conversion Rate
                ROUND(
                    (
                        SELECT SUM(OFCT.RentedOptyCount * 1.0)
                        FROM CHKPI_OPTY_FACT OFCT
                        WHERE OFCT.Corp_Client_Id IN (
                            SELECT Corp_Client_Id 
                            FROM CHKPI_CUSTOMER_DIMENSION 
                            WHERE CustomerName LIKE 'Google%'
                        )
                    ) /
                    NULLIF((
                        SELECT SUM(OFCT.OptyCount * 1.0)
                        FROM CHKPI_OPTY_FACT OFCT
                        WHERE OFCT.Corp_Client_Id IN (
                            SELECT Corp_Client_Id 
                            FROM CHKPI_CUSTOMER_DIMENSION 
                            WHERE CustomerName LIKE 'Google%'
                        )
                    ), 0) * 100, 0
                ) AS ConversionRate

            FROM CHKPI_TENANCY_FACT TF
            JOIN CHKPI_CUSTOMER_DIMENSION CD ON TF.Corp_Client_Id = CD.Corp_Client_Id
            WHERE CD.CustomerName LIKE 'Google%';

        '''
    },
    {
        "question": "What clients have bookings in Singapore?",
        "sql query": '''
            SELECT CD.CustomerName
            FROM CHKPI_TENANCY_FACT TF
            JOIN CHKPI_TENANCY_DIMENSION T ON TF.Cur_Tenancy_Id = T.Cur_Tenancy_Id
            JOIN CHKPI_CUSTOMER_DIMENSION CD ON TF.Corp_Client_Id = CD.Corp_Client_Id
            JOIN CHKPI_GEO_DIMENSION G ON TF.Geo_Id = G.Geo_Id
            JOIN CHKPI_DATE_DIMENSION D ON TF.BookingDate = D.Date
            WHERE G.City = 'Singapore'
            AND D.Date >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1)
            AND D.Date <= GETDATE()
            GROUP BY CD.CustomerName;
        '''
    },
    {
        "question": "What are the top locations by number of move ins so far this year",
        "sql query": '''
            SELECT TOP 10 GD.City, 
                    COUNT(DISTINCT TF.CUR_TENANCY_ID) AS NumberOfMoveIns
            FROM CHKPI_TENANCY_FACT TF
            JOIN CHKPI_GEO_DIMENSION GD ON TF.Geo_Id = GD.Geo_Id
            JOIN CHKPI_DATE_DIMENSION D ON TF.ActualDate = D.Date
            WHERE TF.MoveIns = 'Y' AND D.Year = YEAR(GETDATE())
            AND GD.City <> ''
            GROUP BY GD.City
            ORDER BY NumberOfMoveIns DESC;
        '''
    },
    {
        "question": "Five markets with the highest number of reservations in 2024 for all the clients",
        "sql query": '''
            SELECT    top 5   GD.City,      COUNT(distinct TF.Cur_Tenancy_Id) AS NumberOfReservations  FROM     
            CHKPI_TENANCY_FACT TF  JOIN    
            CHKPI_GEO_DIMENSION GD ON TF.Geo_Id = GD.Geo_Id  JOIN     
                CHKPI_DATE_DIMENSION D ON TF.ActualDate = D.Date 
                WHERE      D.Year = 2024  and GD.City <> '' GROUP BY      GD.City 
                ORDER BY      NumberOfReservations DESC
        '''
    },
    {"question" : "Which of our suppliers has the best conversion rate?",
     "sql query" : '''
                SELECT
                    Top 10
                    SM.Supplier,
                    SUM(SR.BookedCount * 1.0) / count(distinct SR.opti_id)  AS ConversionRate
                FROM CHKPI_RFH_SUPPLIER_REQUEST_RESPONSE SR
                JOIN CHKPI_RFH_SUPPLIER_MASTER SM ON SR.Supplier_Id = SM.Supplier_Id
                JOIN CHKPI_DATE_DIMENSION D ON SR.CreatedDate = D.Date
                WHERE D.Date >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1) AND D.Date <= GETDATE()
                GROUP BY SM.Supplier
                ORDER BY ConversionRate DESC;
        '''
    }
]