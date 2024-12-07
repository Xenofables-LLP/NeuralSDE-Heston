using CSV
using DataFrames

# Path to the option chain CSV file
option_chain_file = "merged_options_with_stock_close.csv"

try
    # Load the CSV file into a DataFrame
    option_chain = CSV.read(option_chain_file, DataFrame)
    println("Option chain file loaded successfully!")
    
    # Display the first few rows of the data
    println("First 5 rows of the data:")
    println(first(option_chain, 5))
    
    # Display basic information about the dataset
    println("\nColumn Names:")
    println(names(option_chain))
    
    println("\nNumber of Rows and Columns:")
    println(size(option_chain))
    
    # Check for missing values in each column
    println("\nMissing Values in Each Column:")
    for col in names(option_chain)
        println("$(col): $(sum(ismissing(option_chain[!, col])))")
    end
catch e
    println("Error: $(e)")
    println("The file could not be loaded. Please check the file path and format.")
end
