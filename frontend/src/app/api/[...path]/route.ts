import { NextRequest, NextResponse } from "next/server";
import { API_BASE } from "@/lib/subnet";

export const dynamic = "force-dynamic";

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const upstream = new URL(`/api/${path.join("/")}`, API_BASE);
  upstream.search = request.nextUrl.search;

  const headers = new Headers();
  const accept = request.headers.get("accept");
  const contentType = request.headers.get("content-type");
  if (accept) headers.set("accept", accept);
  if (contentType) headers.set("content-type", contentType);

  const response = await fetch(upstream, {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.text(),
    cache: "no-store",
  });

  const outHeaders = new Headers();
  for (const key of ["content-type", "cache-control"]) {
    const value = response.headers.get(key);
    if (value) outHeaders.set(key, value);
  }
  return new NextResponse(response.body, { status: response.status, headers: outHeaders });
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}
