import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";
import type { Profile, Message, ProfileFile } from "../types/profile";

const DATA_DIR = path.resolve(process.cwd(), "data/profiles");
const INDEX_FILE = path.join(DATA_DIR, "index.json");

/** Create data/profiles/ if it doesn't exist yet. */
function ensureDir(): void {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
}

/** Read the flat profile list from index.json (no message history). */
function readIndex(): Profile[] {
  ensureDir();
  if (!fs.existsSync(INDEX_FILE)) return [];
  return JSON.parse(fs.readFileSync(INDEX_FILE, "utf-8")) as Profile[];
}

/** Persist the full profile index back to disk. */
function writeIndex(profiles: Profile[]): void {
  ensureDir();
  fs.writeFileSync(INDEX_FILE, JSON.stringify(profiles, null, 2));
}

/** Resolve the per-profile JSON file path from a profile UUID. */
function profilePath(id: string): string {
  return path.join(DATA_DIR, `${id}.json`);
}

/** Return all profiles (no message history). */
export function listProfiles(): Profile[] {
  return readIndex();
}

/** Create a new profile, write its individual JSON file, and add it to the index. */
export function createProfile(name: string): Profile {
  const profile: Profile = {
    id: randomUUID(),
    name,
    createdAt: new Date().toISOString(),
  };
  const file: ProfileFile = { profile, messages: [] };
  ensureDir();
  fs.writeFileSync(profilePath(profile.id), JSON.stringify(file, null, 2));
  const index = readIndex();
  index.push(profile);
  writeIndex(index);
  return profile;
}

/** Return a single profile by UUID, or null if not found. */
export function getProfile(id: string): Profile | null {
  const index = readIndex();
  return index.find((p) => p.id === id) ?? null;
}

/** Return the full message history for a profile (all turns, not windowed). */
export function getHistory(profileId: string): Message[] {
  const fp = profilePath(profileId);
  if (!fs.existsSync(fp)) return [];
  const file = JSON.parse(fs.readFileSync(fp, "utf-8")) as ProfileFile;
  return file.messages;
}

/** Append a single message to the profile's JSON file on disk. */
export function appendMessage(profileId: string, message: Message): void {
  const fp = profilePath(profileId);
  if (!fs.existsSync(fp)) {
    throw new Error(`Profile ${profileId} not found`);
  }
  const file = JSON.parse(fs.readFileSync(fp, "utf-8")) as ProfileFile;
  file.messages.push(message);
  fs.writeFileSync(fp, JSON.stringify(file, null, 2));
}
