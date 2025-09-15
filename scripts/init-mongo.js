// scripts/init-mongo.js
// Initialize MongoDB database and a sample collection

const dbName = 'cicop_docs';
const user = process.env.MONGO_INITDB_ROOT_USERNAME || 'cicop_admin';
const pwd = process.env.MONGO_INITDB_ROOT_PASSWORD || 'cicop_admin_pass';

try {
  db = db.getSiblingDB(dbName);
  db.createCollection('documents');
  db.documents.createIndex({ key: 1 }, { unique: true });
  db.documents.insertOne({ key: 'init', created_at: new Date() });
  print(`Initialized MongoDB database: ${dbName}`);
} catch (e) {
  print(`Mongo init error: ${e}`);
}
