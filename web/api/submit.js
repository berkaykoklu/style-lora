// One finished form, stored as one JSON file.
//
// No database and no account: the person filling this in is doing a favour and
// should not have to sign up for anything. A random suffix on the name means two
// submissions never collide and neither overwrites the other.
import { put } from '@vercel/blob';

const MAX_BYTES = 200_000;

export default async function handler(request, response) {
  if (request.method !== 'POST') {
    return response.status(405).json({ error: 'POST only' });
  }

  const body = request.body;
  if (!body || typeof body !== 'object' || typeof body.answers !== 'object') {
    return response.status(400).json({ error: 'expected {answers, finishedAt}' });
  }

  const serialised = JSON.stringify(body);
  if (serialised.length > MAX_BYTES) {
    return response.status(413).json({ error: 'too large' });
  }

  // The commonest failure by far, and invisible from the outside: the Blob store
  // exists but the deployment predates it, so the token was never injected.
  // Named here rather than left as a generic 500, because the fix is a redeploy
  // and nothing in a generic 500 says so.
  if (!process.env.BLOB_READ_WRITE_TOKEN && !process.env.BLOB_STORE_ID) {
    return response.status(500).json({
      error: 'no blob store connected',
      fix: 'Vercel > Storage > create a Blob store, connect it to this project, then redeploy',
    });
  }

  try {
    const saved = await put(`responses/${Date.now()}.json`, serialised, {
      access: 'public',
      contentType: 'application/json',
      addRandomSuffix: true,
    });
    // The URL is deliberately not returned: nothing downstream needs it, and a
    // page that knows where the answers live is a page that can read them all.
    console.log('stored', saved.pathname);
    return response.status(200).json({ ok: true });
  } catch (error) {
    // The message, not a stack. Whoever is filling this in cannot fix it, but
    // whoever deployed it can, and "could not store" told neither of them
    // anything.
    console.error('store failed', error);
    return response.status(500).json({ error: 'could not store', detail: String(error && error.message || error) });
  }
}
