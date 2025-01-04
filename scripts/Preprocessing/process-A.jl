using CSV
using DataFrames
using Dates

# File paths
options_file = "data/post-no-preference_options_master_option_chain.csv"
stock_file = "data/A.csv"
output_file = "data/options-A-merged.csv"

# Load the CSV files into DataFrames
options_data = CSV.read(options_file, DataFrame)
stock_data = CSV.read(stock_file, DataFrame)

# Filter rows where act_symbol == "A" in the options data
filtered_options = filter(row -> row.act_symbol == "A", options_data)

# Sort stock data by DATE to ensure proper lookup
stock_data_sorted = sort!(stock_data, :DATE)

# Function to find the closest previous date's CLOSE value
function get_previous_close(date, stock_data)
    # Filter stock_data for rows with DATE <= date
    prev_data = filter(row -> row.DATE <= date, stock_data)
    if nrow(prev_data) > 0
        return prev_data[end, :CLOSE]  # Return the CLOSE value of the closest previous date
    else
        return missing  # No previous date found, return missing
    end
end

# Add a CLOSE column to the filtered_options DataFrame
filtered_options.CLOSE = [get_previous_close(row.date, stock_data_sorted) for row in eachrow(filtered_options)]

# Sort the merged dataset by `date` in ascending order (oldest to newest)
sort!(filtered_options, :date)

# Save the resulting DataFrame to a new CSV file
CSV.write(output_file, filtered_options)

println("Filtered, merged, and sorted data with previous CLOSE values saved to: $output_file")
