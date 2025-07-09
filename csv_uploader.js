//build a generic uploader which will take a csv with 4 fields, email, first_name and last name, you will also get a event_id as second paramtere

//for each row we will call an API to register user to that event

const fs = require('fs');

INTERNAL_REG_API = 'https://backend.dev.goldcast.io/core/user/internal_registration/'
ADMIN_TOKEN = "87cf268e-8049-4e54-ab2d-61d67134c1d2"

function parseCsv(csv) {
    // parse the csv and return an array of objects with email, first_name and last_name
    const rows = csv.split('\n').filter(row => row.trim() !== '') // Remove empty rows
    return rows.map(row => {
        // skip first row
        const [first_name, last_name, email] = row.split(',').map(field => field.trim())
        if (first_name === 'first_name') {
            return null
        }
        return { email, first_name, last_name }
    })
}

async function registerUser(email, first_name, last_name, event_id) {
    try {
        const response = await fetch(INTERNAL_REG_API, {
            method: 'POST',
            headers: {
                'Authorization': `Token ${ADMIN_TOKEN}`,
                'Content-Type': 'application/json',
                'Origin': 'https://admin.dev.goldcast.io'
            },
            body: JSON.stringify({ email, first_name, last_name, event_id })
        })
        
        if (!response.ok) {
            response_text = await response.json();
            console.log("response", response_text)
            throw new Error(`HTTP error! status: ${response.status}`)
        }
        
        const result = await response.json()
        console.log(`Successfully registered user: ${email}`)
        return result
    } catch (error) {
        console.error(`Failed to register user ${email}:`, error.message)
        throw error
    }
}

async function handleCsv(csv, event_id) {
    const rows = parseCsv(csv)
    console.log(`Processing ${rows.length} users for event ${event_id}`)
    
    for (const row of rows) {
        try {
            await registerUser(row.email, row.first_name, row.last_name, event_id)
        } catch (error) {
            console.error(`Error processing row:`, row, error.message)
        }
    }
}

// Main execution
async function main() {
    // Read command line arguments
    const args = process.argv.slice(2)
    
    if (args.length !== 2) {
        console.error('Usage: node csv_uploader.js <csv_file_path> <event_id>')
        process.exit(1)
    }
    
    const csvFilePath = args[0]
    const eventId = args[1]
    
    console.log(`Reading CSV from: ${csvFilePath}`)
    console.log(`Event ID: ${eventId}`)
    
    try {
        // Read the CSV file
        const csvContent = fs.readFileSync(csvFilePath, 'utf8')
        console.log('CSV content loaded successfully')
        
        // Process the CSV
        await handleCsv(csvContent, eventId)
        console.log('CSV processing completed')
        
    } catch (error) {
        console.error('Error processing CSV:', error.message)
        process.exit(1)
    }
}

// Run the main function if this script is executed directly
if (require.main === module) {
    main()
}