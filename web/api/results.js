// Every response, for whoever holds the token.
//
// Not open: the answers are a small dataset someone gave their afternoon to, and
// an open endpoint would also let anyone add noise to it by reading the shape
// first. The token lives in an environment variable, never in the page.
import { get, list } from '@vercel/blob';

export default async function handler(request, response) {
  const expected = process.env.RESULTS_TOKEN;
  if (!expected) {
    return response.status(500).json({ error: 'RESULTS_TOKEN is not set' });
  }
  if (request.query.token !== expected) {
    return response.status(401).json({ error: 'bad token' });
  }

  if (!process.env.BLOB_READ_WRITE_TOKEN && !process.env.BLOB_STORE_ID) {
    return response.status(500).json({ error: 'no blob store connected' });
  }

  const { blobs } = await list({ prefix: 'responses/' });

  // A private blob has no fetchable URL: it is read back through the store with
  // this project's own credentials, which is the whole point of it being private.
  const all = await Promise.all(
    blobs.map(async (blob) => {
      const file = await get(blob.pathname, { access: 'private' });
      if (!file || file.statusCode !== 200) {
        return { stored: blob.uploadedAt, error: 'unreadable', pathname: blob.pathname };
      }
      const text = await new Response(file.stream).text();
      return { stored: blob.uploadedAt, ...JSON.parse(text) };
    })
  );

  return response.status(200).json({ count: all.length, responses: all });
}
