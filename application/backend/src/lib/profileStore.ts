import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";
import type { Profile, Message, ProfileFile } from "../types/profile";

const DATA_DIR = path.resolve(process.cwd(), "data/profiles");
const INDEX_FILE = path.join(DATA_DIR, "index.json");

function ensureDir(): void {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
}

function readIndex(): Profile[] {
  ensureDir();
  if (!fs.existsSync(INDEX_FILE)) return [];
  return JSON.parse(fs.readFileSync(INDEX_FILE, "utf-8")) as Profile[];
}

function writeIndex(profiles: Profile[]): void {
  ensureDir();
  fs.writeFileSync(INDEX_FILE, JSON.stringify(profiles, null, 2));
}

function profilePath(id: string): string {
  return path.join(DATA_DIR, `${id}.json`);
}

export function listProfiles(): Profile[] {
  return readIndex();
}

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

export function getProfile(id: string): Profile | null {
  const index = readIndex();
  return index.find((p) => p.id === id) ?? null;
}

export function getHistory(profileId: string): Message[] {
  const fp = profilePath(profileId);
  if (!fs.existsSync(fp)) return [];
  const file = JSON.parse(fs.readFileSync(fp, "utf-8")) as ProfileFile;
  return file.messages;
}

export function appendMessage(profileId: string, message: Message): void {
  const fp = profilePath(profileId);
  if (!fs.existsSync(fp)) {
    throw new Error(`Profile ${profileId} not found`);
  }
  const file = JSON.parse(fs.readFileSync(fp, "utf-8")) as ProfileFile;
  file.messages.push(message);
  fs.writeFileSync(fp, JSON.stringify(file, null, 2));
}
