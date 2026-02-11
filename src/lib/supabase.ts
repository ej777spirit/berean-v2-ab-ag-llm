import { createClient, SupabaseClient } from '@supabase/supabase-js';

// SUPABASE CLIENT (lazy-initialized for SSR safety)

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

let _supabase: SupabaseClient | null = null;

function getSupabase(): SupabaseClient | null {
  if (!supabaseUrl || !supabaseAnonKey) return null;
  if (!_supabase) {
    _supabase = createClient(supabaseUrl, supabaseAnonKey);
  }
  return _supabase;
}

export const supabase = { get client() { return getSupabase(); } };

export function isSupabaseConfigured(): boolean {
  return Boolean(supabaseUrl && supabaseAnonKey);
}

// KEY-VALUE STORAGE

export async function loadData<T>(key: string, fallback: T): Promise<T> {
  if (!isSupabaseConfigured()) {
    try {
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : fallback;
    } catch {
      return fallback;
    }
  }
  try {
    const { data, error } = await getSupabase()!
      .from('berean_store')
      .select('value')
      .eq('key', key)
      .single();
    if (error || !data) return fallback;
    return data.value as T;
  } catch {
    return fallback;
  }
}

export async function saveData<T>(key: string, value: T): Promise<void> {
  if (!isSupabaseConfigured()) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
      console.error('Storage error:', e);
    }
    return;
  }
  try {
    const { error } = await getSupabase()!
      .from('berean_store')
      .upsert(
        { key, value: JSON.stringify(value) },
        { onConflict: 'key' }
      );
    if (error) {
      console.error('Supabase save error:', error);
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch (e) {
        console.error('Fallback storage error:', e);
      }
    }
  } catch (e) {
    console.error('Storage error:', e);
  }
}

// TYPED DATA OPERATIONS

export interface SequenceRecord {
  id?: string;
  name: string;
  vh_sequence?: string;
  vl_sequence?: string;
  species?: string;
  isotype?: string;
  source?: string;
  notes?: string;
  binding_results?: Record<string, unknown>;
  cdr_regions?: Record<string, unknown>;
  developability?: Record<string, unknown>;
}

export interface CandidateRecord {
  id?: string;
  name: string;
  sequence_id?: string;
  avg_binding?: number;
  breadth_score?: number;
  developability_score?: number;
  stage?: string;
  binding_profile?: Record<string, unknown>;
  notes?: string;
}

export interface LiteratureRecord {
  id?: string;
  title: string;
  authors?: string;
  journal?: string;
  year?: number;
  doi?: string;
  pmid?: string;
  abstract?: string;
  tags?: string[];
  notes?: string;
}

export const db = {
  sequences: {
    async getAll() {
      if (!isSupabaseConfigured()) return [];
      const { data } = await getSupabase()!.from('sequences').select('*').order('created_at', { ascending: false });
      return data || [];
    },
    async upsert(record: SequenceRecord) {
      if (!isSupabaseConfigured()) return null;
      const { data } = await getSupabase()!.from('sequences').upsert(record).select().single();
      return data;
    },
    async delete(id: string) {
      if (!isSupabaseConfigured()) return;
      await getSupabase()!.from('sequences').delete().eq('id', id);
    }
  },
  candidates: {
    async getAll() {
      if (!isSupabaseConfigured()) return [];
      const { data } = await getSupabase()!.from('candidates').select('*').order('created_at', { ascending: false });
      return data || [];
    },
    async upsert(record: CandidateRecord) {
      if (!isSupabaseConfigured()) return null;
      const { data } = await getSupabase()!.from('candidates').upsert(record).select().single();
      return data;
    },
    async delete(id: string) {
      if (!isSupabaseConfigured()) return;
      await getSupabase()!.from('candidates').delete().eq('id', id);
    }
  },
  literature: {
    async getAll() {
      if (!isSupabaseConfigured()) return [];
      const { data } = await getSupabase()!.from('literature').select('*').order('created_at', { ascending: false });
      return data || [];
    },
    async insert(record: LiteratureRecord) {
      if (!isSupabaseConfigured()) return null;
      const { data } = await getSupabase()!.from('literature').insert(record).select().single();
      return data;
    },
    async delete(id: string) {
      if (!isSupabaseConfigured()) return;
      await getSupabase()!.from('literature').delete().eq('id', id);
    }
  }
};
